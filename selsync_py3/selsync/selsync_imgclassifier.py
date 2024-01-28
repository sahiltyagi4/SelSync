import logging
import socket
import time
import os
import argparse

import torch
import torch.distributed as dist

import selsync_py3.helper.helper_fns as misc
import selsync_py3.helper.dnn_models as models
import selsync_py3.helper.selsync_datapartitioner as seldp
import selsync_py3.helper.noniid_datapartitioner as noniid_dp
from selsync_py3.helper.datainjection import DataInjection


class SelSyncImgClassifier(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.rank = args.rank
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.backend = args.backend
        self.logdir = args.dir
        self.model_name = args.model
        self.data_dist = args.data_dist
        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-'
                                     + str(self.rank)+ '.log', level=logging.INFO)
        dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
        self.dataset_name = args.dataset
        if args.determinism == 0:
            self.determinism = False
        elif args.determinism == 1:
            self.determinism = True

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        if self.data_dist == 'iid':
            self.dataset_obj = seldp.SelsyncDatasets(dataset_name=self.dataset_name, args=args)
        elif self.data_dist == 'noniid':
            self.dataset_obj = noniid_dp.NonIIDDataset(dataset_name=self.dataset_name, args=args)
            args.labels_per_rank = self.dataset_obj.get_labelsperrank()

        if args.data_injection == 0:
            self.inject_data = False
        else:
            self.inject_data = True
            self.datainjection = DataInjection(args=args, dataset=self.dataset_name, valid_labels=args.labels_per_rank)

        self.lr_gamma = args.gamma
        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.globalstep = 0

        self.trainloader = self.dataset_obj.get_trainloader()
        self.testloader = self.dataset_obj.get_testloader()
        self.train_loss, self.topaccs = None, None
        self.train_steps = args.trainsteps
        self.epochs = args.epochs
        self.test_steps = args.teststeps
        self.windowsize = args.windowsize
        self.smoothing = args.smoothing
        self.smoothing_method = args.smoothing_method
        # print per-layer gradnorms at this frequency
        self.layer_printnorm = args.layer_printnorm

        self.comm_ops = misc.CollectiveCommunicationOps(world_size=self.world_size)
        # track gradient norm
        self.local_modelstats = misc.TrackModelStats(model=self.model, windowsize=self.windowsize,
                                                     smoothing=self.smoothing, method=self.smoothing_method)

        self.delta_ewma = misc.EWMAMeter(windowsize=self.windowsize, alpha=self.smoothing)

        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        args.datapartition = self.dataset_obj.__class__.__name__

        args.dataset_size = self.dataset_obj.get_trainsize()
        self.dataset_size = args.dataset_size
        self.delta_threshold = args.delta_threshold
        self.aggregation = args.aggregation

        logging.info(f'model arguments are {args}')

    def test_accuracy(self, curr_epoch, prefix='STEP'):
        with torch.no_grad():
            if self.globalstep >= self.test_steps and self.globalstep % self.test_steps == 0:
                top1acc, top5acc, top10acc = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
                top100acc, top1000acc = misc.AverageMeter(), misc.AverageMeter()
                test_loss = misc.AverageMeter()
                for input, label in self.testloader:
                    input, label = input.to(self.device), label.to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label)
                    # compute the top-1, top-5, top-10 accuracy.
                    if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
                        topaccs = misc.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10))
                        top1acc.update(topaccs[0], input.size(0))
                        top5acc.update(topaccs[1], input.size(0))
                        top10acc.update(topaccs[2], input.size(0))
                        test_loss.update(loss.item(), input.size(0))

                    elif self.dataset_name == 'imagenet':
                        topaccs = misc.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10, 100, 1000))
                        top1acc.update(topaccs[0], input.size(0))
                        top5acc.update(topaccs[1], input.size(0))
                        top10acc.update(topaccs[2], input.size(0))
                        test_loss.update(loss.item(), input.size(0))
                        top100acc.update(topaccs[3], input.size(0))
                        top1000acc.update(topaccs[4], input.size(0))

                if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
                    logging.info(
                        f'{prefix} VALIDATION METRICS step {self.globalstep} epoch {curr_epoch} testlossval {test_loss.val} '
                        f'testlossavg {test_loss.avg} top1val {top1acc.val.cpu().numpy().item()} '
                        f'top1avg {top1acc.avg.cpu().numpy().item()} top5val {top5acc.val.cpu().numpy().item()} '
                        f'top5avg {top5acc.avg.cpu().numpy().item()} top10val {top10acc.val.cpu().numpy().item()} '
                        f'top10avg {top10acc.avg.cpu().numpy().item()}')

                elif self.dataset_name == 'imagenet':
                    logging.info(
                        f'{prefix} VALIDATION METRICS step {self.globalstep} epoch {curr_epoch} testlossval {test_loss.val} '
                        f'testlossavg {test_loss.avg} top1val {top1acc.val.cpu().numpy().item()} '
                        f'top1avg {top1acc.avg.cpu().numpy().item()} top5val {top5acc.val.cpu().numpy().item()} '
                        f'top5avg {top5acc.avg.cpu().numpy().item()} top10val {top10acc.val.cpu().numpy().item()} '
                        f'top10avg {top10acc.avg.cpu().numpy().item()} top100val {top100acc.val.cpu().numpy().item()} '
                        f'top100avg {top100acc.avg.cpu().numpy().item()} top1000val {top1000acc.val.cpu().numpy().item()} '
                        f'top1000avg {top1000acc.avg.cpu().numpy().item()}')

    def train_accuracy(self, input, label, output, loss, epoch):
        with torch.no_grad():
            if self.globalstep >= self.train_steps and self.globalstep % self.train_steps == 0:

                if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
                    trainaccs = misc.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10))
                    self.top1accs.update(trainaccs[0], input.size(0))
                    self.top5accs.update(trainaccs[1], input.size(0))
                    self.top10accs.update(trainaccs[2], input.size(0))
                    self.train_loss.update(loss.item(), input.size(0))

                    logging.info(
                        f'TRAINING METRICS step {self.globalstep} epoch {epoch} trainlossval {self.train_loss.val} '
                        f'trainlossavg {self.train_loss.avg} top1val {self.top1accs.val.cpu().numpy().item()} '
                        f'top1avg {self.top1accs.avg.cpu().numpy().item()} top5val {self.top5accs.val.cpu().numpy().item()} '
                        f'top5avg {self.top5accs.avg.cpu().numpy().item()} top10val {self.top10accs.val.cpu().numpy().item()} '
                        f'top10avg {self.top10accs.avg.cpu().numpy().item()}')

                elif self.dataset_name == 'imagenet':
                    trainaccs = misc.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10, 100, 1000))
                    self.top1accs.update(trainaccs[0], input.size(0))
                    self.top5accs.update(trainaccs[1], input.size(0))
                    self.top10accs.update(trainaccs[2], input.size(0))
                    self.train_loss.update(loss.item(), input.size(0))
                    self.top100accs.update(trainaccs[3], input.size(0))
                    self.top1000accs.update(trainaccs[4], input.size(0))

                    logging.info(
                        f'TRAINING METRICS step {self.globalstep} epoch {epoch} trainlossval {self.train_loss.val} '
                        f'trainlossavg {self.train_loss.avg} top1val {self.top1accs.val.cpu().numpy().item()} '
                        f'top1avg {self.top1accs.avg.cpu().numpy().item()} top5val {self.top5accs.val.cpu().numpy().item()} '
                        f'top5avg {self.top5accs.avg.cpu().numpy().item()} top10val {self.top10accs.val.cpu().numpy().item()} '
                        f'top10avg {self.top10accs.avg.cpu().numpy().item()} top100val {self.top100accs.val.cpu().numpy().item()} '
                        f'top100avg {self.top100accs.avg.cpu().numpy().item()} top1000val '
                        f'{self.top1000accs.val.cpu().numpy().item()} top1000avg {self.top1000accs.avg.cpu().numpy().item()}')

    def start_training(self):
        self.model = self.comm_ops.broadcast(model=self.model, rank=0)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        processed_samples = 0
        last_epoch = None
        prev_gradnorm, curr_gradnorm = 1., 1.

        for _ in range(self.epochs):
            self.top1accs, self.top5accs, self.top10accs = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            self.train_loss = misc.AverageMeter()
            self.top100accs, self.top1000accs = misc.AverageMeter(), misc.AverageMeter()
            for record in self.trainloader:
                input, label = record

                if self.inject_data:
                    input, label = self.datainjection.inject_data(input, label)

                input, label = input.to(self.device), label.to(self.device)
                self.globalstep += 1

                begin = time.time()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                compute_time = time.time() - begin

                gradients = [p.grad for p in self.model.parameters()]
                curr_gradnorm = self.local_modelstats.compute_gradnorm(gradients=gradients)
                del gradients

                curr_delta = abs(curr_gradnorm - prev_gradnorm) / prev_gradnorm
                self.delta_ewma.smoothendata(curr_delta)
                smooth_delta = self.delta_ewma.smooth_val() if self.globalstep >= self.windowsize else None

                if smooth_delta is None or smooth_delta >= self.delta_threshold:
                    signal_tensor = torch.LongTensor([1])
                else:
                    signal_tensor = torch.LongTensor([0])

                signal_tensor = self.comm_ops.tensor_reduce(tensor=signal_tensor)
                # if signal_tensor > 0, perform allreduce, else do local update
                if int(signal_tensor.item()) > 0:
                    method = 'SYNC'
                    processed_samples += self.train_bsz * self.world_size
                    if self.aggregation == 'parameter':
                        # parameter aggregation among workers
                        begin = time.time()
                        self.model = self.comm_ops.allreduce_modelparams(self.model)
                        sync_time = time.time() - begin
                    elif self.aggregation == 'gradient':
                        # gradient aggregation among workers
                        begin = time.time()
                        self.model = self.comm_ops.allreduce_grads(self.model)
                        sync_time = time.time() - begin

                elif int(signal_tensor.item()) == 0:
                    method = 'LOCAL'
                    processed_samples += self.train_bsz
                    sync_time = None

                if self.lr_scheduler is not None:
                    sched_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    sched_lr = self.opt.param_groups[0]['lr']

                opt_lr = self.opt.param_groups[0]['lr']
                self.opt.step()
                self.opt.zero_grad()
                epoch = processed_samples // self.dataset_size

                logging.info(f'logging iteration step {self.globalstep} epoch {epoch} curr_gradnorm {curr_gradnorm} '
                             f'prev_gradnorm {prev_gradnorm} compute_time {compute_time} sync_time {sync_time} '
                             f'sched_lr {sched_lr} opt_lr {opt_lr} smooth_delta {smooth_delta} '
                             f'signal_tensor {signal_tensor.item()} selsync_method {method}')
                prev_gradnorm = curr_gradnorm

                self.train_accuracy(input=input, label=label, output=output, loss=loss, epoch=epoch)
                self.test_accuracy(curr_epoch=epoch, prefix='STEP')

                if self.lr_scheduler is not None and self.model_name == 'resnet101' and epoch != last_epoch:
                    if epoch == 110 or epoch == 150 or epoch == 190 or epoch == 250:
                        self.lr_scheduler.get_last_lr()[0] = self.opt.param_groups[0]['lr'] * self.lr_gamma
                        self.opt.param_groups[0]['lr'] *= self.lr_gamma
                        last_epoch = epoch
                        logging.info(f'last_epoch logged {last_epoch} curr_epoch {epoch}')
                elif self.lr_scheduler is not None and self.model_name == 'vgg11' and epoch != last_epoch:
                    if epoch == 50 or epoch == 75 or epoch == 100:
                        self.lr_scheduler.get_last_lr()[0] = self.opt.param_groups[0]['lr'] * self.lr_gamma
                        self.opt.param_groups[0]['lr'] *= self.lr_gamma
                        last_epoch = epoch
                        logging.info(f'last_epoch logged {last_epoch} curr_epoch {epoch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--dir', type=str, default='/', help='dir where data is saved')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--test-bsz', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet101')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--determinism', type=int, default=0)
    parser.add_argument('--trainsteps', type=int, default=100)
    parser.add_argument('--teststeps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data-injection', type=int, default=0, help='disable (0) or enable (1) data-injection')
    parser.add_argument('--alpha', type=float, default=0.25, help='data-injection param for cluster-size')
    parser.add_argument('--beta', type=float, default=0.25, help='data-injection parameter for sample sharing')

    # either simple or exp or custom_ewma
    parser.add_argument('--smoothing-method', type=str, default='custom_ewma')
    parser.add_argument('--delta-threshold', type=float, default=0.3)
    parser.add_argument('--data-dist', type=str, default='iid')
    parser.add_argument('--aggregation', type=str, default='parameter')
    parser.add_argument('--datapartition', type=str, default='selsyncpartition')
    parser.add_argument('--train-dir', type=str, default='/')
    parser.add_argument('--test-dir', type=str, default='/')
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--slope-window', type=int, default=100)

    #parameters for ewm on gradient norm
    parser.add_argument('--windowsize', type=int, default=10)
    parser.add_argument('--smoothing', type=float, default=0.001)
    parser.add_argument('--layer-printnorm', type=int, default=100)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
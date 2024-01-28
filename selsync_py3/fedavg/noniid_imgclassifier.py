import logging
import socket
import time
import os
import argparse

import torch
import torch.distributed as dist

import selsync_py3.helper.helper_fns as misc
import selsync_py3.helper.dnn_models as models
import selsync_py3.helper.noniid_datapartitioner as noniid_data


class NonIID_FedAvgImgClassifier(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.rank = args.rank
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.backend = args.backend
        self.logdir = args.dir
        self.model_name = args.model
        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-' + str(self.rank)
                                     + '.log', level=logging.INFO)
        tcp_addr = 'tcp://' + str(args.master_addr) + ':' + str(args.master_port)
        import datetime
        timeout = datetime.timedelta(seconds=3000 * 60 * 60 * 100)
        dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=self.rank, world_size=self.world_size,
                                timeout=timeout)

        self.dataset_name = args.dataset
        if args.determinism == 0:
            self.determinism = False
        elif args.determinism == 1:
            self.determinism = True

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        self.dataset_obj = noniid_data.NonIIDDataset(dataset_name=self.dataset_name, args=args)

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
        self.comm_ops = misc.CollectiveCommunicationOps(world_size=self.world_size)
        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        args.datapartition = self.dataset_obj.__class__.__name__
        args.dataset_size = self.dataset_obj.get_trainsize()
        self.dataset_size = args.dataset_size
        args.method = 'FedAvg'
        self.fedavg_steps = args.fedavg_steps

        logging.info(f'model arguments are {args}')

    def test_accuracy(self, curr_epoch, prefix='STEP'):
        with torch.no_grad():
            if self.globalstep >= self.test_steps and self.globalstep % self.test_steps == 0:
                top1acc, top5acc, top10acc = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
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

                if self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
                    logging.info(
                        f'{prefix} VALIDATION METRICS step {self.globalstep} epoch {curr_epoch} testlossval {test_loss.val} '
                        f'testlossavg {test_loss.avg} top1val {top1acc.val.cpu().numpy().item()} '
                        f'top1avg {top1acc.avg.cpu().numpy().item()} top5val {top5acc.val.cpu().numpy().item()} '
                        f'top5avg {top5acc.avg.cpu().numpy().item()} top10val {top10acc.val.cpu().numpy().item()} '
                        f'top10avg {top10acc.avg.cpu().numpy().item()}')

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

    def start_training(self):
        self.model = self.comm_ops.broadcast(model=self.model, rank=0)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        processed_samples = 0
        last_epoch = None

        for e in range(self.epochs):
            self.top1accs, self.top5accs, self.top10accs = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            self.train_loss = misc.AverageMeter()
            for record in self.trainloader:
                itr_strt = time.time()
                input, label = record
                input, label = input.to(self.device), label.to(self.device)
                self.globalstep += 1

                begin = time.time()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                compute_time = time.time() - begin

                if self.globalstep >= self.fedavg_steps and self.globalstep % self.fedavg_steps == 0:
                    begin = time.time()
                    self.model = self.comm_ops.allreduce_modelparams(self.model)
                    processed_samples += self.train_bsz * self.world_size
                    sync_mode = 'SYNC'
                    sync_time = time.time() - begin
                else:
                    sync_time = None
                    sync_mode = 'LOCAL'
                    processed_samples += self.train_bsz

                epoch = processed_samples // self.dataset_size
                if self.lr_scheduler is not None:
                    sched_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    sched_lr = self.opt.param_groups[0]['lr']

                opt_lr = self.opt.param_groups[0]['lr']
                begin = time.time()
                self.opt.step()
                self.opt.zero_grad()
                sgdupdate_time = time.time() - begin
                itr_time = time.time() - itr_strt

                logging.info(f'logging iteration step {self.globalstep} fed e {e} epoch {epoch} '
                             f'compute_time {compute_time} sync_time {sync_time} '
                             f'sgdupdate_time {sgdupdate_time} iteration_time {itr_time} sched_lr {sched_lr} '
                             f'opt_lr {opt_lr} sync_model {sync_mode}')

                self.train_accuracy(input=input, label=label, output=output, loss=loss, epoch=epoch)
                self.test_accuracy(curr_epoch=epoch, prefix='STEP')

                if self.lr_scheduler is not None and self.model_name == 'resnet101' and epoch != last_epoch:
                    if epoch == 110 or epoch == 150 or epoch == 190 or epoch == 250:
                        self.lr_scheduler.get_last_lr()[0] = self.opt.param_groups[0]['lr'] * self.lr_gamma
                        self.opt.param_groups[0]['lr'] *= self.lr_gamma
                        last_epoch = epoch
                        logging.info(f'last_epoch logged {last_epoch} curr_epoch {epoch} and e {e}')
                elif self.lr_scheduler is not None and self.model_name == 'vgg11' and epoch != last_epoch:
                    if epoch == 50 or epoch == 75 or epoch == 100:
                        self.lr_scheduler.get_last_lr()[0] = self.opt.param_groups[0]['lr'] * self.lr_gamma
                        self.opt.param_groups[0]['lr'] *= self.lr_gamma
                        last_epoch = epoch
                        logging.info(f'last_epoch logged {last_epoch} curr_epoch {epoch} and e {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=10)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--backend', type=str, default='gloo', help='use mpi, gloo or nccl')
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
    parser.add_argument('--fedavg-steps', type=int, default=250)
    parser.add_argument('--step-size', type=int, default=5)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    NonIID_FedAvgImgClassifier(args).start_training()
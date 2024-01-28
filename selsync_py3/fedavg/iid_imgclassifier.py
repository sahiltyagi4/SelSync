import logging
import socket
import time
import math
import os
import argparse

import torch
import torch.distributed as dist

import selsync_py3.helper.helper_fns as misc
import selsync_py3.helper.dnn_models as models
from selsync_py3.helper.default_datapartitioner import Dataset


class FedAvgImageClassifier(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.rank = args.rank
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.backend = args.backend
        self.logdir = args.dir
        self.model_name = args.model
        logging.basicConfig(
            filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-' + str(self.rank) + '.log',
            level=logging.INFO)
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

        self.dataset_obj = Dataset(dataset_name=self.dataset_name, args=args)
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
        args.method = 'FedAvg'
        self.fedavg_steps = args.fedavg_steps

        args.dataset_size = self.dataset_obj.get_trainsize()
        self.dataset_size = args.dataset_size
        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        args.datapartition = self.dataset_obj.__class__.__name__
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
                    # computes top-1, top-5, top-10 accuracy.
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

        for e in range(self.epochs):
            self.top1accs, self.top5accs, self.top10accs = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            self.train_loss = misc.AverageMeter()
            self.top100accs, self.top1000accs = misc.AverageMeter(), misc.AverageMeter()
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

                if self.lr_scheduler is not None:
                    sched_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    sched_lr = self.opt.param_groups[0]['lr']

                opt_lr = self.opt.param_groups[0]['lr']
                begin = time.time()
                self.opt.step()
                self.opt.zero_grad()
                sgdupdate_time = time.time() - begin

                if self.globalstep >= self.fedavg_steps and self.globalstep % self.fedavg_steps == 0:
                    # Instead of aggregating gradients, we aggregate model parameters in FedAvg
                    begin = time.time()
                    self.model = self.comm_ops.allreduce_modelparams(self.model)
                    sync_time = time.time() - begin
                    sync_mode = 'SYNC'
                else:
                    sync_time = None
                    sync_mode = 'LOCAL'

                processed_samples += self.train_bsz
                epoch = math.ceil(processed_samples / self.dataset_size)

                itr_time = time.time() - itr_strt

                logging.info(f'logging iteration step {self.globalstep} fed e {e} epoch {epoch} '
                             f'compute_time {compute_time} sync_time {sync_time} '
                             f'sgdupdate_time {sgdupdate_time} iteration_time {itr_time} sched_lr {sched_lr} '
                             f'opt_lr {opt_lr} sync_model {sync_mode}')

                # compute train and test accuracy after certain number of iterations
                self.train_accuracy(input=input, label=label, output=output, loss=loss, epoch=epoch)
                self.test_accuracy(curr_epoch=epoch, prefix='STEP')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=10)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--backend', type=str, default='gloo', help='use mpi, gloo or nccl')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--dir', type=str, default='/', help='dir where data is saved')
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--test-bsz', type=int, default=10)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--determinism', type=int, default=0)
    parser.add_argument('--trainsteps', type=int, default=100)
    parser.add_argument('--teststeps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=5.0)
    parser.add_argument('--fedavg-steps', type=int, default=250)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr-decay-steps', type=int, default=5000)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    FedAvgImageClassifier(args).start_training()
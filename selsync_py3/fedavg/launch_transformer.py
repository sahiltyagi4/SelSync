import logging
import socket
import math
import time
import os
import argparse

import torch
from torch import Tensor
import torch.distributed as dist

import selsync_py3.helper.helper_fns as misc
import selsync_py3.helper.dnn_models as models
import selsync_py3.helper.default_datapartitioner as dp


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class FedAvgTransformer(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.rank = args.rank
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.backend = args.backend
        self.logdir = args.dir
        self.model_name = args.model
        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-'
                                     + str(self.rank) + '.log',level=logging.INFO)
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

        self.norm_clip = args.norm_clip
        self.dataset_obj = dp.Dataset(dataset_name=self.dataset_name, args=args)
        self.trainloader = self.dataset_obj.get_trainloader()
        self.testloader = self.dataset_obj.get_testloader()
        # get_trainsize returns vocab in wikitext2 dataset
        self.vocab = self.dataset_obj.get_vocab()
        args.ntoken = len(self.vocab)
        self.ntokens = args.ntoken
        print(f'# of tokens {self.ntokens}')

        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.bptt = args.bptt
        self.globalstep = 0

        self.train_loss, self.topaccs = None, None
        self.train_steps = args.trainsteps
        self.epochs = args.epochs
        self.test_steps = args.teststeps
        self.lr_decay_steps = args.lr_decay_steps

        self.comm_ops = misc.CollectiveCommunicationOps(world_size=self.world_size)
        args.method = 'FedAvg'
        self.fedavg_steps = args.fedavg_steps

        args.dataset_size = self.dataset_obj.get_trainsize()
        self.dataset_size = args.dataset_size
        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        args.datapartition = self.dataset_obj.__class__.__name__
        logging.info(f'model arguments are {args}')

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

    def test_loss(self, data_source, curr_epoch, prefix='STEP'):
        with torch.no_grad():
            if self.globalstep > self.test_steps and self.globalstep % self.test_steps == 0:
                src_mask = generate_square_subsequent_mask(self.bptt).to(self.device)
                test_loss = 0.
                for i in range(0, data_source.size(0) - 1, self.bptt):
                    data, targets = self.get_batch(data_source, i)
                    data, targets = data.to(self.device), targets.to(self.device)
                    if data.size(0) != self.bptt:
                        src_mask = generate_square_subsequent_mask(data.size(0)).to(self.device)

                    output = self.model(data, src_mask)
                    flat_output = output.view(-1, self.ntokens)
                    test_loss += len(data) * self.loss(flat_output, targets).item()

                mean_testloss = test_loss / (len(data_source) - 1)
                logging.info(f'{prefix} VALIDATION METRICS step {self.globalstep} epoch {curr_epoch} test_loss_mean '
                             f'{mean_testloss} test_perplexity_mean {math.exp(mean_testloss)}')

    def start_training(self):
        self.model = self.comm_ops.broadcast(model=self.model, rank=0)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        processed_samples = 0

        for e in range(self.epochs):
            src_mask = generate_square_subsequent_mask(self.bptt).to(self.device)
            total_loss = 0.

            for batch, i in enumerate(range(0, self.trainloader.size(0) - 1, self.bptt)):
                itr_strt = time.time()
                data, targets = self.get_batch(self.trainloader, i)
                data, targets = data.to(self.device), targets.to(self.device)
                if data.size(0) != self.bptt:
                    src_mask = src_mask[:data.size(0), :data.size(0)].to(self.device)

                self.globalstep += 1
                begin = time.time()
                output = self.model(data, src_mask)
                loss = self.loss(output.view(-1, self.ntokens), targets)
                total_loss += loss.item()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
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

                sync_time = None
                sync_mode = 'LOCAL'
                processed_samples += self.train_bsz
                epoch = math.ceil(processed_samples / self.dataset_size)

                itr_time = time.time() - itr_strt

                if self.globalstep >= self.train_steps and self.globalstep % self.train_steps == 0:
                    curr_loss = total_loss / self.train_steps
                    train_perplexity = math.exp(curr_loss)
                    logging.info(f'TRAINING METRICS step {self.globalstep} epoch {epoch} train_loss {total_loss} '
                                 f'train_perplexity {train_perplexity} sched_lr {sched_lr} opt_lr {opt_lr}')
                    total_loss = 0.

                logging.info(f'logging iteration step {self.globalstep} fed e {e} epoch {epoch} '
                             f'compute_time {compute_time} sync_time {sync_time} '
                             f'sgdupdate_time {sgdupdate_time} iteration_time {itr_time} sched_lr {sched_lr} '
                             f'opt_lr {opt_lr} sync_model {sync_mode}')

                self.test_loss(data_source=self.testloader, curr_epoch=epoch, prefix='STEP')

                if self.globalstep >= self.lr_decay_steps and self.globalstep % self.lr_decay_steps == 0 and self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            begin = time.time()
            self.model = self.comm_ops.allreduce_modelparams(self.model)
            sync_time = time.time() - begin
            logging.info(f'aggregated model at end of epoch {epoch} step {self.globalstep} sync_time {sync_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=8)
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
    parser.add_argument('--d-model', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--d-hid', type=int, default=200)
    parser.add_argument('--step-size', type=float, default=1.0)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--norm-clip', type=float, default=0.5)
    parser.add_argument('--lr-decay-steps', type=int, default=5000)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    FedAvgTransformer(args).start_training()
import logging
import socket
import math
import time

import torch
from torch import Tensor
import torch.distributed as dist

import selsync_py3.helper.helper_fns as misc
import selsync_py3.helper.dnn_models as models
import selsync_py3.helper.selsync_datapartitioner as seldp
import selsync_py3.helper.default_datapartitioner as defdp


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class SelSyncTransformer(object):
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
                                     + str(self.rank) + '.log', level=logging.INFO)

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
        self.datapartition = args.datapartition

        if self.datapartition == 'selsyncpartition':
            self.dataset_obj = seldp.SelsyncDatasets(dataset_name=self.dataset_name, args=args)

        elif self.datapartition == 'defaultpartition':
            self.dataset_obj = defdp.Dataset(dataset_name=self.dataset_name, args=args)

        self.trainloader = self.dataset_obj.get_trainloader()
        self.testloader = self.dataset_obj.get_testloader()
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
        self.slope_window = args.slope_window

        self.train_loss, self.topaccs = None, None
        self.train_steps = args.trainsteps
        self.epochs = args.epochs
        self.test_steps = args.teststeps
        self.windowsize = args.windowsize
        self.smoothing = args.smoothing
        self.smoothing_method = args.smoothing_method
        # print per-layer gradnorms at this frequency
        self.layer_printnorm = args.layer_printnorm
        self.lr_decay_steps = args.lr_decay_steps

        self.comm_ops = misc.CollectiveCommunicationOps(world_size=self.world_size)
        # prior gradient aggregation statistics
        self.local_modelstats = misc.TrackModelStats(model=self.model, windowsize=self.windowsize,
                                                     smoothing=self.smoothing, method=self.smoothing_method)

        self.delta_ewma = misc.EWMAMeter(windowsize=self.windowsize, alpha=self.smoothing)
        self.plot_grads = misc.PlotGradientDistributions(param_names=self.local_modelstats.param_names)

        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        args.datapartition = self.dataset_obj.__class__.__name__

        args.dataset_size = self.dataset_obj.get_trainsize()
        self.dataset_size = args.dataset_size
        self.delta_threshold = args.delta_threshold
        self.aggregation = args.aggregation

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
        prev_gradnorm, curr_gradnorm = 1., 1.

        for _ in range(self.epochs):
            src_mask = generate_square_subsequent_mask(self.bptt).to(self.device)
            total_loss = 0.

            for batch, i in enumerate(range(0, self.trainloader.size(0) - 1, self.bptt)):
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
                else:
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

                logging.info(f'logging iteration step {self.globalstep} epoch {epoch} curr_gradnorm '
                             f'{curr_gradnorm} prev_gradnorm {prev_gradnorm} compute_time {compute_time} '
                             f'sync_time {sync_time} sched_lr {sched_lr} opt_lr {opt_lr} smooth_delta {smooth_delta} '
                             f'signal_tensor {signal_tensor.item()} selsync_method {method}')
                prev_gradnorm = curr_gradnorm

                if self.globalstep >= self.train_steps and self.globalstep % self.train_steps == 0:
                    curr_loss = total_loss / self.train_steps
                    train_perplexity = math.exp(curr_loss)
                    logging.info(f'TRAINING METRICS step {self.globalstep} epoch {epoch} train_loss {total_loss} '
                                 f'train_perplexity {train_perplexity} sched_lr {sched_lr} opt_lr {opt_lr}')
                    total_loss = 0.

                self.test_loss(data_source=self.testloader, curr_epoch=epoch, prefix='STEP')

                if self.globalstep >= self.lr_decay_steps and self.globalstep % self.lr_decay_steps == 0 and self.lr_scheduler is not None:
                    self.lr_scheduler.step()
import numpy as np
import random
import math
import os
import logging
from _random import Random
from prettytable import PrettyTable
from numpy import linspace
from scipy.stats.kde import gaussian_kde

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


def set_seed(seed, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(determinism)


def compute_imgaccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value for model loss, accuracy etc."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EWMAMeter(object):
    def __init__(self, windowsize=25, alpha=0.01):
        self.windowsize = windowsize
        self.alpha = alpha
        self.window = []
        self.ewma_val = 0.

    def smoothendata(self, val):
        sum_val = 0.
        count = 0.
        self.window.append(val)
        if len(self.window) == self.windowsize:
            for index, datum in reversed(list(enumerate(self.window))):
                pow_ix = len(self.window) - index
                sum_val += math.pow((1 - self.alpha), pow_ix) * datum
                count += math.pow((1 - self.alpha), pow_ix)

            self.ewma_val = sum_val / count
            self.window.pop(0)

    def smooth_val(self):
        return self.ewma_val


class TrackModelStats(object):
    def __init__(self, model, windowsize, smoothing, method):
        self.param_names = []
        self.param_shapes = {}
        self.param_lengths = {}
        self.requires_grad = {}
        self.method = method
        # for each layer
        self.smooth_meter = {}
        self.model_smoothmeter = EWMAMeter(windowsize=windowsize, alpha=smoothing)

        for name, param in model.named_parameters():
            self.requires_grad[name] = param.requires_grad
            self.param_names.append(name)
            self.param_shapes[name] = list(param.size())
            self.param_lengths[name] = param.numel()
            self.smooth_meter[name] = EWMAMeter(windowsize=windowsize, alpha=smoothing)

        table = PrettyTable(["Module", "Parameters"])
        total_params = 0
        ctr = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
            ctr += 1

        print(table)
        print(f"counter is {ctr}")
        print(f"Total Trainable Params: {total_params}")
        total_size = (total_params * 4) / (1024 * 1024)
        print(f"Model memory footprint using single precision: {total_size} MB")

    def compute_gradnorm(self, gradients):
        total_gnorm = 0.
        for name, g in zip(self.param_names, gradients):
            # process ewm grad norm for each layer separately
            layer_gnorm = torch.norm(g.flatten())
            self.smooth_meter[name].smoothendata(layer_gnorm)
            total_gnorm += layer_gnorm

        self.model_smoothmeter.smoothendata(total_gnorm)
        # return total_gnorm
        return self.model_smoothmeter.smooth_val()


class CollectiveCommunicationOps(object):
    def __init__(self, world_size):
        self.world_size = world_size

    def tensor_reduce(self, tensor):
        dist.all_reduce(tensor=tensor, op=ReduceOp.SUM)
        return tensor

    # average and synchronize gradients
    def allreduce_grads(self, model):
        for _, param in model.named_parameters():
            dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM)
            param.grad = param.grad / self.world_size

        return model

    # average and synchronize model parameters
    def allreduce_modelparams(self, model):
        for _, param in model.named_parameters():
            dist.all_reduce(tensor=param.data, op=ReduceOp.SUM)
            param.data = param.data / self.world_size

        return model

    def broadcast(self, model, rank=0):
        for _, param in model.named_parameters():
            if not param.requires_grad: continue
            dist.broadcast(tensor=param.data, src=rank)

        return model


class PlotWeightDistributions(object):
    def __init__(self, param_names):
        import matplotlib.pyplot as plt
        self.param_names = param_names
        plt.rcParams.update({'font.size': 16})
        plt.style.use('seaborn-whitegrid')

    def create_directory(self, logdir, step, rank):
        stepdir = 'weightstep-' + str(step) + '-' + str(rank)
        # added so that plots are created inside per-rank log directory
        ranklogdir = os.path.join(logdir, 'g' + str(rank))
        os.mkdir(os.path.join(ranklogdir, stepdir))
        return os.path.join(ranklogdir, stepdir), os.path.exists(os.path.join(ranklogdir, stepdir))

    def plot_weightdist(self, weights, stepdir):
        size = len(self.param_names)
        for ix in range(0, size):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            plt.gcf().subplots_adjust(bottom=0.2, left=0.15)
            plt.style.use('seaborn-whitegrid')
            name = self.param_names[ix].replace('.', '_')
            weight_tensor = weights[ix].view(-1).to(torch.device("cpu"))
            kde = gaussian_kde(weight_tensor)
            x_dist = linspace(min(weight_tensor), max(weight_tensor), 100)
            plt.plot(x_dist, kde(x_dist), label=self.param_names[ix], color='k')
            plt.xlabel('Weights')
            plt.grid('on', axis='y', linestyle='--', alpha=0.6)
            plt.grid('on', axis='x', linestyle='--', alpha=0.6)
            plt.savefig(stepdir + '/' + f'{name}.pdf')
            plt.close(fig)


class PlotGradientDistributions(object):
    def __init__(self, param_names):
        import matplotlib.pyplot as plt
        self.param_names = param_names
        plt.rcParams.update({'font.size': 16})

    def create_directory(self, logdir, step, rank):
        stepdir = 'trainstep-' + str(step) + '-' + str(rank)
        # added so that plots are created inside each worker rank's log directory
        ranklogdir = os.path.join(logdir, 'g'+str(rank))
        os.mkdir(os.path.join(ranklogdir, stepdir))
        return os.path.join(ranklogdir, stepdir), os.path.exists(os.path.join(ranklogdir, stepdir))

    # plots kernel density estimates of the gradients
    def plot_gradients(self, gradients, stepdir):
        size = len(self.param_names)

        for ix in range(0, size):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            plt.gcf().subplots_adjust(bottom=0.2, left=0.15)
            name = self.param_names[ix].replace('.','_')
            g_tensor = gradients[ix].view(-1).to(torch.device("cpu"))
            kde = gaussian_kde(g_tensor)
            x_dist = linspace(min(g_tensor), max(g_tensor), 100)
            plt.plot(x_dist, kde(x_dist), label=self.param_names[ix], color='k')
            plt.xlabel('Gradient')
            plt.savefig(stepdir + '/' + f'{name}.pdf')
            plt.close(fig)


# to compute the eigen values of the Hessian
class EigenValueComposition(object):
    def __init__(self, power_itr_count, top_n_eigenvals):
        self.power_itr_count = power_itr_count
        self.top_n_eigenvals = top_n_eigenvals

    def power_iteration(self, curr_step, model, grad_vec, random_vec):
        random_vec = random_vec / torch.norm(random_vec)
        prev_lambda = 0.
        prev_vec = random_vec
        for i in range(0, self.power_itr_count):
            hv_vector_dict = torch.autograd.grad(grad_vec, model.parameters(), grad_outputs=prev_vec,
                                                 only_inputs=True, retain_graph=True)
            hessian_vec = torch.cat([hv.contiguous().view(-1) for hv in hv_vector_dict])
            curr_vec = hessian_vec.detach()
            curr_vec = curr_vec / torch.norm(curr_vec)
            curr_lambda = prev_vec.dot(curr_vec).item()
            repeat_lambda = sum([torch.sum(p * q) for (p,q) in zip(prev_vec, curr_vec)])
            if curr_lambda == 0.:
                error = 1.
            else:
                error = abs((curr_lambda - prev_lambda) / curr_lambda)
            prev_lambda = curr_lambda
            prev_vec = curr_vec.detach()
            eigenvec_norm = torch.norm(curr_vec.contiguous().view(-1))
            logging.info(f'eigen_compute globalstep {curr_step} power_itr_count {i} error {error} '
                         f'eigen_val {curr_lambda} repeat_lab {repeat_lambda} eigenvec_norm {eigenvec_norm}')

        return curr_lambda, curr_vec

    def power_itr(self, curr_step, model, grad_vec):
        v = torch.cat([torch.randn_like(p.grad.contiguous().view(-1)) for p in model.parameters()])
        v = v / (torch.norm(v) + 1e-8)
        for i in range(self.power_itr_count):
            gv = torch.inner(torch.t(grad_vec), v)
            hv_vector_dict = torch.autograd.grad(gv, model.parameters(), create_graph=True)
            hv_vector = torch.cat([hv.contiguous().view(-1) for hv in hv_vector_dict])
            v = hv_vector / (torch.norm(hv_vector) + 1e-8)

        logging.info(f'selsync eigenvalues step {curr_step} eigen_val {gv.item()} eigen_vec {v}')
import math
import random

import torch

import selsync_py3.helper.noniid_datapartitioner as noniid_data


class DataInjection(object):
    def __init__(self, args, dataset, valid_labels):
        if dataset == 'cifar10':
            self.total_labels = 10
            self.perlabel_dataloader = noniid_data.perlabel_loaderCIFAR10(train_dir=args.dir, bsz=args.bsz,
                                                                          input_shape=[3, 32, 32], out_shape=[1],
                                                                          seed=args.seed, determinism=args.determinism,
                                                                          num_classes=self.total_labels)
        elif dataset == 'cifar100':
            self.total_labels = 100
            self.perlabel_dataloader = noniid_data.perlabel_loaderCIFAR100(train_dir=args.dir, bsz=args.bsz,
                                                                           input_shape=[3, 32, 32], out_shape=[1],
                                                                           seed=args.seed, determinism=args.determinism,
                                                                           num_classes=self.total_labels)

        self.rank = args.rank
        self.valid_labels = valid_labels
        self.alpha = args.alpha
        self.beta = args.beta
        self.injected_classes = int(math.ceil(self.alpha * self.total_labels))
        self.effective_bsz = (args.beta * args.bsz * args.world_size ) // self.total_labels

    def inject_data(self, input, label):
        random_label = -1
        for _ in range(self.injected_classes):
            while random_label in self.valid_labels:
                random_label = random.randint(0, self.total_labels -1)

            added_data = self.perlabel_dataloader[random_label]
            added_input, added_label = next(iter(added_data))
            added_bsz = added_input.size()[0]

            if self.effective_bsz < added_bsz:
                added_input, added_label = added_input[:self.effective_bsz], added_label[:self.effective_bsz]

            input = torch.cat((input, added_input), dim=0)
            label = torch.cat((label, added_label), dim=0)

        return (input, label)
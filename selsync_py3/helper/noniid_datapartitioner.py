import math
import logging

import torch
import torchvision
import torchvision.transforms as transforms

import selsync_py3.helper.helper_fns as misc


def cifar10_partitionedlabels(seed, train_dir, train_bsz, num_classes=10, input_shape=[3, 32, 32], out_shape=[1]):
    perlabel_dataloaders = {}
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform)
    for i in range(num_classes):
        record = []
        for rec in train_data:
            if rec[1] == i:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                rec = (inp, lab)
                record.append(rec)

        trainloader = torch.utils.data.DataLoader(record, batch_size=train_bsz, shuffle=True)
        perlabel_dataloaders[i] = trainloader

    del train_data

    return perlabel_dataloaders


def cifar100_partitionedlabels(seed, train_dir, train_bsz, num_classes=100, input_shape=[3, 32, 32], out_shape=[1]):
    perlabel_dataloader = {}
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform)
    for i in range(num_classes):
        record = []
        for rec in train_data:
            if rec[1] == i:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                rec = (inp, lab)
                record.append(rec)

        trainloader = torch.utils.data.DataLoader(record, batch_size=train_bsz, shuffle=True)
        perlabel_dataloader[i] = trainloader

    del train_data

    return perlabel_dataloader


class NonIIDDataset(object):
    def __init__(self, dataset_name, args):
        if dataset_name == 'cifar10':
            self.trainloader, self.trainsize, self.labels_per_rank = nonIID_CIFAR10(train_dir=args.dir, seed=args.seed,
                                                                                    rank=args.rank, world_size=args.world_size,
                                                                                    total_labels=10, train_bsz=args.bsz,
                                                                                    num_workers=1)
            self.testloader = cifar10_test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)
        elif dataset_name == 'cifar100':
            self.trainloader, self.trainsize, self.labels_per_rank = nonIID_CIFAR100(train_dir=args.dir, seed=args.seed,
                                                                                     rank=args.rank, world_size=args.world_size,
                                                                                     total_labels=100, train_bsz=args.bsz,
                                                                                     num_workers=1)
            self.testloader = cifar100_test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)

    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return self.testloader

    def get_trainsize(self):
        return self.trainsize

    def get_labelsperrank(self):
        return self.labels_per_rank


def get_labels_per_rank(rank, world_size, total_labels):
    # assumes labels are partitioned evenly among workers, i.e., world_size is a multiple of total labels
    labels_perworker = total_labels // world_size
    labels_per_rank = []
    for i in range(0, labels_perworker):
        labels_per_rank.append(labels_perworker * rank + i)

    return labels_per_rank


def nonIID_CIFAR10(train_dir, seed, rank, world_size, total_labels, train_bsz, num_workers=1):

    labels_per_rank = get_labels_per_rank(rank, world_size, total_labels)
    logging.info(f'labels_per_rank {labels_per_rank}')
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])

    train_data = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    records = []
    for input, label in loader:
        if label[0].item() in labels_per_rank:
            input_label = input, label
            records.append(input_label)

    del loader
    train_size = len(records)

    recs = None
    input_shape = [1, 3, 32, 32]
    output_shape = [1]
    for i,l in records:
        i = i.reshape(input_shape)
        l = l.reshape(output_shape)
        if recs is not None:
            i = torch.concat([recs[0], i], dim=0)
            l = torch.concat([recs[1], l], dim=0)

        recs = i, l

    perlabel_dataset = torch.utils.data.TensorDataset(recs[0], recs[1])

    trainloader = torch.utils.data.DataLoader(perlabel_dataset, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    del records

    return trainloader, train_size, labels_per_rank


def cifar10_test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)

    return testloader


def nonIID_CIFAR100(train_dir, seed, rank, world_size, total_labels, train_bsz, num_workers=1):

    labels_per_rank = get_labels_per_rank(rank, world_size, total_labels)
    logging.info(f'labels_per_rank {labels_per_rank}')
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    records = []
    for input, label in loader:
        if label[0].item() in labels_per_rank:
            input_label = input, label
            records.append(input_label)

    del loader
    train_size = len(records)

    recs = None
    input_shape = [1, 3, 32, 32]
    output_shape = [1]
    for i, l in records:
        i = i.reshape(input_shape)
        l = l.reshape(output_shape)
        if recs is not None:
            i = torch.concat([recs[0], i], dim=0)
            l = torch.concat([recs[1], l], dim=0)

        recs = i, l

    perlabel_dataset = torch.utils.data.TensorDataset(recs[0], recs[1])
    trainloader = torch.utils.data.DataLoader(perlabel_dataset, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    del records

    return trainloader, train_size, labels_per_rank


def cifar100_test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR100(root=log_dir, train=False,
                                            download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)

    return testloader


def perlabel_loaderCIFAR10(train_dir, bsz, input_shape, out_shape, seed, determinism, num_classes):
    perlabel_dataloader = {}
    misc.set_seed(seed, determinism=determinism)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True, download=True, transform=transform)
    for i in range(0,num_classes):
        record = []
        for rec in train_data:
            if rec[1] == i:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                rec = (inp, lab)
                record.append(rec)

        trainloader = torch.utils.data.DataLoader(record, batch_size=bsz, shuffle=True)
        perlabel_dataloader[i] = trainloader

    del train_data
    print(f'size of perlabel loader map {len(perlabel_dataloader)}')
    return perlabel_dataloader


def perlabel_loaderCIFAR100(train_dir, bsz, input_shape, out_shape, seed, determinism, num_classes):
    perlabel_dataloader = {}

    misc.set_seed(seed, determinism=determinism)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    train_data = torchvision.datasets.CIFAR100(root=train_dir + 'data', train=True, download=True, transform=transform)

    for c in range(0, num_classes):
        record = []
        for rec in train_data:
            if rec[1] == c:
                inp = torch.tensor(rec[0]).reshape(input_shape).to(torch.float32)
                lab = torch.scalar_tensor(rec[1]).reshape(out_shape).to(torch.long)
                record.append((inp, lab))

        trainloader = torch.utils.data.DataLoader(record, batch_size=bsz, shuffle=True)
        perlabel_dataloader[c] = trainloader

    return perlabel_dataloader
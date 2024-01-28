import random
import logging

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import Tensor

import selsync_py3.helper.helper_fns as misc


class SelsyncDatasets(object):
    def __init__(self, dataset_name, args):
        if dataset_name == 'cifar10':
            self.trainloader, self.train_size = selsync_cifar10Train(log_dir=args.dir, world_size=args.world_size,
                                                                 trainer_rank=args.rank, train_bsz=args.bsz,
                                                                 seed=args.seed, num_workers=1)
            self.testloader = cifar10Test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'cifar100':
            self.trainloader, self.train_size = selsync_cifar100Train(log_dir=args.dir, world_size=args.world_size,
                                                                  trainer_rank=args.rank, train_bsz=args.bsz,
                                                                  seed=args.seed, num_workers=1)
            self.testloader = cifar100Test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'imagenet':
            self.trainloader, self.train_size = selsync_imagenetTrain(train_dir=args.train_dir, bsz=args.bsz, seed=args.seed,
                                                                  world_size=args.world_size, trainer_rank=args.rank)
            self.testloader = imagenetTest(test_dir=args.test_dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'wikitext':
            self.trainloader, self.testloader, self.vocab, self.train_size = selsync_wikitextData(bsz=args.bsz, test_bsz=args.test_bsz,
                                                                                                  trainer_rank=args.rank, seed=args.seed,
                                                                                                  world_size=args.world_size)


    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return self.testloader

    def get_trainsize(self):
        return self.train_size

    def get_vocab(self):
        return self.vocab


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class SelsyncDataPartioner(object):
    def __init__(self, data, rank, world_size):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.seed(rank)
        random.shuffle(indexes)
        logging.info(f'for rank {rank} top-10 ixs are {indexes[0:10]}')

        partitions = [1. for _ in range(0, world_size)]
        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def selsync_cifar10Train(log_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    trainset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=True,
                                            download=True, transform=transform)
    partition = SelsyncDataPartioner(trainset, trainer_rank, world_size)
    partition = partition.use(trainer_rank)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g, num_workers=num_workers)

    return trainloader, len(trainset)


def cifar10Test(log_dir, test_bsz, seed):
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


def selsync_cifar100Train(log_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,4), normalize])
    trainset = torchvision.datasets.CIFAR100(root=log_dir, train=True,
                                             download=True, transform=transform)
    partition = SelsyncDataPartioner(trainset, trainer_rank, world_size)
    partition = partition.use(trainer_rank)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    return trainloader, len(trainset)


def cifar100Test(log_dir, test_bsz, seed):
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


def selsync_imagenetTrain(train_dir, bsz, world_size, seed, trainer_rank):

    misc.set_seed(seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size = (224, 256)
    dataset = datasets.ImageFolder(train_dir, transforms.Compose([transforms.RandomSizedCrop(size[0]),
                                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                                  normalize]))
    partition = SelsyncDataPartioner(dataset, trainer_rank, world_size)
    partition = partition.use(trainer_rank)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, worker_init_fn=misc.set_seed(seed),
                                              generator=g, num_workers=1)
    return trainloader, len(trainloader) * bsz


def imagenetTest(test_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size[1]), transforms.CenterCrop(size[0]),
                                    transforms.ToTensor(), normalize])
    testloader = torch.utils.data.DataLoader(datasets.ImageFolder(test_dir, transform=transform),
                                            batch_size=test_bsz, shuffle=False, generator=g,
                                            num_workers=4)
    return testloader


def selsync_wikitextData(train_bsz, test_bsz, trainer_rank, seed, world_size):
    misc.set_seed(seed)
    train_iter = WikiText103(split='train')

    print(f'trainiter type {type(train_iter)}')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText103()
    train_data = data_process(train_iter)
    test_data = data_process(test_iter)

    partition = SelsyncDataPartioner(data=train_data, rank=trainer_rank, world_size=world_size)
    partition = partition.use(trainer_rank)

    def training_batches(training_data, bsz):
        seq_len = len(training_data) // bsz
        training_data = training_data[:seq_len * bsz]
        training_data = training_data.view(bsz, seq_len).t().contiguous()
        return training_data

    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.
        Args:
            data: Tensor, shape [N]
            bsz: int, batch size
        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data

    train_data = training_batches(partition, train_bsz)  # shape [seq_len, batch_size]
    test_data = batchify(test_data, test_bsz)

    # return train_data, test_data, vocab
    return train_data, test_data, vocab, len(train_data)
import math

import torch
from torchvision import models
from torch import nn, optim, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder

import selsync_py3.helper.helper_fns as misc


def get_model(model_name, determinism, args):
    if model_name == 'resnet101':
        model_obj = ResNet101Obj(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed,
                              gamma=args.gamma, determinism=determinism)

    elif model_name == 'alexnet':
        model_obj = AlexNetObj(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed,
                            gamma=args.gamma, determinism=determinism, step_size=args.step_size)

    elif model_name == 'transformer':
        model_obj = TransformerObj(ntoken=args.ntoken, d_model=args.d_model, dropout=args.dropout, nhead=args.nhead,
                                   nlayers=args.nlayers, d_hid=args.d_hid, lr=args.lr, gamma=args.gamma, seed=args.seed,
                                   determinism=determinism, step_size=args.step_size)

    elif model_name == 'vgg11':
        model_obj = VGG11Obj(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed,
                             determinism=determinism, gamma=args.gamma)

    return model_obj


class VGG11Obj(object):
    def __init__(self, lr, momentum, seed, weight_decay, gamma, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.vgg11(pretrained=False, progress=True)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[50, 75, 100],
                                                                 gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class ResNet101Obj(object):
    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.resnet101(progress=True, pretrained=False)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weightdecay)
        milestones = [110, 150, 190, 250]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=milestones, gamma=self.gamma,
                                                           last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class AlexNetObj(object):
    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism, step_size):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.step_size = step_size
        self.model = models.alexnet(progress=True, pretrained=False)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = None

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class TransformerObj(object):

    def __init__(self, ntoken, d_model, dropout, nhead, nlayers, d_hid, lr, gamma, seed, determinism, step_size):
        misc.set_seed(seed=seed, determinism=determinism)
        self.model = Transformer(ntoken=ntoken, d_model=d_model, dropout=dropout, nhead=nhead, nlayers=nlayers,
                                 d_hid=d_hid)
        self.lr = lr
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size, gamma=gamma)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
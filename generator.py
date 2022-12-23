import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

def init(obj, mean=0.0, std=0.01):
    classname = obj.__class__.__name__
    if classname.find("Conv") != -1:
        obj.weight.data.normal_(mean, std)

def pad(k, d=1):
    return int((k - 1) * d / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, c, k=3, d=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(c, c, k, 1, dilation=d[0], padding=pad(k, d[0]))),
            weight_norm(Conv1d(c, c, k, 1, dilation=d[1], padding=pad(k, d[1]))),
            weight_norm(Conv1d(c, c, k, 1, dilation=d[2], padding=pad(k, d[2])))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1))),
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1))),
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1)))
        ])

        self.convs1.apply(init)
        self.convs2.apply(init)
    
    def remove_weight_norm(self):
        for _ in self.convs1:
            remove_weight_norm(_)
        for _ in self.convs2:
            remove_weight_norm(_)

    def forward(self, x):
        output = x
        for c1, c2 in zip(self.convs1, self.convs2):
            out = c1(F.leaky_relu(output, 0.1))
            out = c2(F.leaky_relu(out, 0.1))
            output = out + output
        return output


class Generator(torch.nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.kernels = len(params['resblock_kernel_sizes'])
        self.samples = len(params['upsample_rates'])
        self.pre = weight_norm(Conv1d(80, params['upsample_initial_channel'], 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(params['upsample_rates'], params['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(
                ConvTranspose1d(params['upsample_initial_channel'] // (2 ** i), params['upsample_initial_channel'] // (2 ** (i + 1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = params['upsample_initial_channel'] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(params['resblock_kernel_sizes'], params['resblock_dilation_sizes'])):
                self.resblocks.append(ResBlock(ch, k, d))

        self.post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init)
        self.post.apply(init)

    def remove_weight_norm(self):
        for _ in self.ups:
            remove_weight_norm(_)
        for _ in self.resblocks:
            _.remove_weight_norm()
        remove_weight_norm(self.pre)
        remove_weight_norm(self.post)

    def forward(self, x):
        output = self.pre(x)
        for i in range(self.samples):
            output = self.ups[i](F.leaky_relu(output, 0.1))
            out = None
            for j in range(self.kernels):
                if out is None:
                    out = self.resblocks[i * self.kernels + j](output)
                else:
                    out += self.resblocks[i * self.kernels + j](output)
            output = out / self.kernels
        output = torch.tanh(self.post(F.leaky_relu(output)))
        return output

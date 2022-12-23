import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


def init(obj, mean=0.0, std=0.01):
    classname = obj.__class__.__name__
    if classname.find("Conv") != -1:
        obj.weight.data.normal_(mean, std)

def pad(k, d=1):
    return int((k - 1) * d / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, h, c, k=3, dil=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convblock1 = nn.ModuleList([
            weight_norm(Conv1d(c, c, k, 1, dilation=dil[0], padding=pad(k, dil[0]))),
            weight_norm(Conv1d(c, c, k, 1, dilation=dil[1], padding=pad(k, dil[1]))),
            weight_norm(Conv1d(c, c, k, 1, dilation=dil[2], padding=pad(k, dil[2])))
        ])
        self.convblock2 = nn.ModuleList([
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1))),
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1))),
            weight_norm(Conv1d(c, c, k, 1, dilation=1, padding=pad(k, 1)))
        ])
        
        self.convs1.apply(init)
        self.convs2.apply(init)

    def forward(self, x):
        output = x.copy()
        for conv1, conv2 in zip(self.convblock1, self.convblock2):
            out = nn.functional.leaky_relu(output, 0.1)
            out = conv1(out)
            out = nn.functional.leaky_relu(out, 0.1)
            out = conv2(out)
            output += out 
        return output

    def remove_weight_norm(self):
        for _ in self.convblock1:
            remove_weight_norm(_)
        for _ in self.convblock2:
            remove_weight_norm(_)


class Generator(torch.nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        self.conv_pre = weight_norm(Conv1d(80, conf['upsample_initial_channel'], 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(conf['upsample_rates'], conf['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(
                ConvTranspose1d(conf['upsample_initial_channel']//(2 ** i), conf['upsample_initial_channel'] // (2 ** (i + 1)),
                                k, u, padding=(k - u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = conf['upsample_initial_channel'] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(conf['resblock_kernel_sizes'], conf['resblock_dilation_sizes'])):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init)
        self.conv_post.apply(init)
        
        self.config = conf
        self.num_kernels = len(conf['resblock_kernel_sizes'])
        self.num_upsamples = len(conf['upsample_rates'])

    def forward(self, x):
        out = self.conv_pre(x)
        for i in range(self.num_upsamples):
            out = F.leaky_relu(out, 0.1)
            out = self.ups[i](out)
            res = None
            for j in range(self.num_kernels):
                if res is None:
                    res = self.resblocks[i * self.num_kernels + j](out)
                else:
                    res += self.resblocks[i * self.num_kernels + j](out)
            out = res / self.num_kernels
        output = torch.tanh(self.conv_post(nn.functional.leaky_relu(out)))
        return output

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for _ in self.ups:
            remove_weight_norm(_)
        for _ in self.resblocks:
            _.remove_weight_norm()
        remove_weight_norm(self.conv_post)

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


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

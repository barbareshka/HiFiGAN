import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

class PD(torch.nn.Module):
    def __init__(self, p, k=5, s=3, spec=False):
        super(PD, self).__init__()
        self.p = p
        norm_f = weight_norm 
        if spec:
            norm_f = spectral_norm
        self.conv = nn.ModuleList([
            norm_f(Conv2d(1, 32, (k, 1), (s, 1), padding=(2, 0))),
            norm_f(Conv2d(32, 128, (k, 1), (s, 1), padding=(2, 0))),
            norm_f(Conv2d(128, 512, (k, 1), (s, 1), padding=(2, 0))),
            norm_f(Conv2d(512, 1024, (k, 1), (s, 1), padding=(2, 0))),
            norm_f(Conv2d(1024, 1024, (k, 1), 1, padding=(2, 0))),
        ])
        self.last = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # prepare vec
        d1, d2, d3 = x.shape
        if d3 % self.p != 0:
            pad = self.p - (d3 % self.p)
            x = F.pad(x, (0, pad), "reflect")
            d3 = d3 + pad
        prep = x.view(d1, d2, d3 // self.p, self.p)

        vecs = []
        for layer in self.conv:
            prep = F.leaky_relu(layer(prep), 0.1)
            vecs.append(prep)

        out = self.last(prep)
        vecs.append(out)
        out = torch.flatten(out, 1, -1)
        return out, vecs


class MPD(torch.nn.Module):
    def __init__(self):
        super(MPD, self).__init__()
        self.discriminators = nn.ModuleList([PD(2), PD(3), PD(5), PD(7), PD(11)])

    def forward(self, y, y_hat):
        rs = []
        gs = []
        vecs_r = []
        vecs_g = []
        for i, disc in enumerate(self.discriminators):
            r, vec_r = disc(y)
            rs.append(r)
            vecs_r.append(vec_r)

            g, vec_g = disc(y_hat)
            gs.append(g)
            vecs_g.append(vec_g)

        return rs, gs, vecs_r, vecs_g


class SD(torch.nn.Module):
    def __init__(self, spec=False):
        super(SD, self).__init__()
        norm_f = weight_norm 
        if spec:
            norm_f = spectral_norm
        self.conv = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.last = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        out = x
        vecs = []
        for layer in self.conv:
            out = F.leaky_relu(layer(out), 0.1)
            vecs.append(x)
        out = self.last(out)
        vecs.append(out)
        out = torch.flatten(out, 1, -1)
        return x, vecs


class MSD(torch.nn.Module):
    def __init__(self):
        super(MSD, self).__init__()
        self.discriminators = nn.ModuleList([SD(True), SD(), SD()])
        self.pools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        rs = []
        gs = []
        vecs_r = []
        vecs_g = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                y = self.pools[i-1](y)
                y_hat = self.pools[i-1](y_hat)
            r, vec_r = disc(y)
            rs.append(r)
            vecs_r.append(vec_r)

            g, vec_g = disc(y_hat)
            gs.append(g)
            vecs_g.append(vec_g)

        return rs, gs, vecs_r, vecs_g

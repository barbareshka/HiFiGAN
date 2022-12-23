import torch

def d_loss(real, gen):
    total = 0.
    reals = []
    gens = []
    for e1, e2 in zip(real, gen):
        l_r = torch.mean((1 - e1) ** 2)
        l_g = torch.mean(e2 ** 2)
        reals.append(l_r.item())
        gens.append(l_g.item())
        total += (l_r + l_g)
    return total, reals, gens


def g_loss(res):
    total = 0.
    losses = []
    for _ in res:
        loss = torch.mean((1 - _) ** 2)
        losses.append(loss)
        total += loss
    return total, losses


def f_loss(real, gen):
    total = 0.
    for e1, e2 in zip(real, gen):
        for r, g in zip(e1, e2):
            total += torch.mean(torch.abs(r - g))
    total *= 2
    return total

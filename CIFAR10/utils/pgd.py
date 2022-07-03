import torch
from torch.nn import functional as F


def pgd_inf(model, data, target, epsilon=8/255, alpha=2/255, steps=10, random_start=True):
    training_mode = model.training
    model.eval()
    delta = torch.zeros_like(data, requires_grad=True)
    if random_start:
        delta.data.uniform_(-epsilon, epsilon)
    for _ in range(steps):
        with torch.enable_grad():
            loss = F.cross_entropy(model(delta+data), target)
        grad = torch.autograd.grad(loss, [delta])[0].data
        delta.data = delta.data + alpha*torch.sign(grad)
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(data.data+delta.data, 0, 1) - data.data
    adv_data = delta.data + data.data
    model.train(training_mode)
    return adv_data


def pgd_2(model, data, target, epsilon=1.5, alpha=0.2, steps=10, rand_init=False):
    training_mode = model.training
    model.eval()
    batch_lenth = len(target)
    delta = torch.zeros_like(data, requires_grad=True)
    if rand_init:
        delta.data.normal_()
        norm = torch.linalg.norm(delta.data.view(batch_lenth, -1), ord=2, dim=1) + 1e-9
        rand_r = torch.empty_like(norm).uniform_()
        delta.data *= (rand_r/norm).reshape(-1, 1, 1, 1)

    for _ in range(steps):
        with torch.enable_grad():
            loss = F.cross_entropy(model(delta+data), target)
        grad = torch.autograd.grad(loss, [delta])[0].data
        grad_norm = torch.linalg.norm(grad.view(batch_lenth, -1), ord=2, dim=1).reshape(-1, 1, 1, 1) + 1e-9
        grad = grad/grad_norm

        delta.data = delta.data + alpha*grad
        delta_norm = torch.linalg.norm(delta.data.view(batch_lenth, -1), ord=2, dim=1) + 1e-9
        factor = torch.min(torch.ones_like(delta_norm), epsilon/delta_norm).reshape(-1, 1, 1, 1)
        delta.data = delta.data * factor
        delta.data = torch.clamp(delta.data+data.data, 0, 1) - data.data

    model.train(training_mode)
    adv_data = data.data + delta.data
    return adv_data

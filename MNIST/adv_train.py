from networks import lenet
from data import get_train_loader, get_test_loader
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import os
from utils import *
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('CIFAR10-PGD-AT')
    parser.add_argument('--adv-train', default=True)
    parser.add_argument('--batch-size', default=64, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.02, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=(100, 105), type=tuple[int])
    parser.add_argument('--gamma', default=0.1, type=float)
    return parser.parse_args()


def pgd_at():
    save_path = os.path.join(path, 'ckpts')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_state = np.array([-1, 0.9, 0.8])  # (-best_loss, best_acc, best_adv_acc)

    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        total = 0
        for data, target in train_loader:
            total += len(data)
            data, target = data.to(device), target.to(device)
            if args.adv_train:
                data = pgd_inf(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            loss = F.cross_entropy(model(data), target)
            optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.item()*len(data))
            optimizer.step()
        scheduler.step()
        avg_loss = np.array(loss_list).sum().item()/total
        test_acc, test_adv_acc = pgd_test(test_loader)
        logger.add_scalar('avg loss', avg_loss, global_step=epoch)
        logger.add_scalar('acc', test_acc, global_step=epoch)
        logger.add_scalar('adv acc', test_adv_acc, global_step=epoch)
        if epoch >= args.milestones[0]:
            temp_state = np.array([-avg_loss, test_acc, test_adv_acc])
            cur_state = np.max((temp_state, best_state), axis=0)
            if (cur_state != best_state).any():
                torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pth'))
                best_state = cur_state
    torch.save(model.state_dict(), os.path.join(save_path, 'end.pth'))


def pgd_test(loader):
    model.eval()
    total = 0
    correct = 0
    adv_correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        total += len(data)
        with torch.no_grad():
            correct += model(data).argmax(1).eq(target).sum().item()
            adv_data = pgd_inf(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            adv_correct += model(adv_data).argmax(1).eq(target).sum().item()
    acc = correct/total
    adv_acc = adv_correct/total
    return acc, adv_acc


if __name__ == '__main__':
    device = 'cuda:3'
    args = get_args()
    test_loader = get_test_loader(args.batch_size)
    train_loader = get_train_loader(args.batch_size)
    model = lenet().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    path = os.path.join('logs', 'lenet')
    os.makedirs(path, exist_ok=True)
    logger = SummaryWriter(logdir=path)
    logger.add_text('args', str(args))
    pgd_at()

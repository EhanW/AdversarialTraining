from networks import *
from data import get_train_loader, get_test_loader
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import os
from utils import *
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('CIFAR10-PGD-ADV-DISTILLATION')
    parser.add_argument('--adv-distill', default=True)
    parser.add_argument('--proxy_model_name', default='resnet18', type=str, choices=all_model_names)
    parser.add_argument('--target_model_name', default='resnet18', type=str, choices=all_model_names)
    parser.add_argument('--target_ckpt_path', default='logs/resnet18/ckpts/end.pth', type=str)
    parser.add_argument('--temperature', default=5, type=int)
    parser.add_argument('--kd-lambda', default=0.5, type=int)
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--epsilon', default=8/255, type=float)
    parser.add_argument('--alpha', default=2/255, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=(100, 105), type=tuple[int])
    parser.add_argument('--gamma', default=0.1, type=float)
    return parser.parse_args()


def distill():
    save_path = os.path.join(path, 'ckpts')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        proxy_model.train()
        loss_list = []
        total = 0
        for data, target in train_loader:
            total += len(data)
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                if args.adv_distill:
                    data = pgd_inf(proxy_model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
                target_logits = target_model(data)
            proxy_logits = proxy_model(data)

            loss_gt = F.cross_entropy(proxy_logits, target)
            loss_kd = F.kl_div(F.log_softmax(proxy_logits / args.temperature, dim=1),
                               F.softmax(target_logits / args.temperature, dim=1), reduction='batchmean')
            loss = args.kd_lambda*loss_gt + (1-args.kd_lambda)*loss_kd

            loss_list.append(loss.item()*len(data))

            proxy_optimizer.zero_grad()
            loss.backward()
            proxy_optimizer.step()
        proxy_scheduler.step()
        avg_loss = np.array(loss_list).sum().item()/total
        acc, adv_acc, transfer_adv_acc = test(test_loader)

        #print(
        #    f'Epoch[{epoch + 1}/110], train loss:{avg_loss:.3f}, '
        #    f'acc:{acc:.3f}, self adv acc:{adv_acc:.3f}, transfer adv acc:{transfer_adv_acc:.3f}.'
        #)
        logger.add_scalar('avg loss', avg_loss, global_step=epoch)
        logger.add_scalar('acc', acc, global_step=epoch)
        logger.add_scalar('adv acc', adv_acc, global_step=epoch)
        logger.add_scalar('transfer adv acc', transfer_adv_acc, global_step=epoch)

    torch.save(proxy_model.state_dict(), os.path.join(save_path, 'end.pth'))


def test(loader):
    proxy_model.eval()
    correct = 0
    total = 0
    self_adv_correct = 0
    transfer_adv_correct = 0
    for data, target in loader:
        total += len(data)
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            proxy_logits = proxy_model(data)
            correct += proxy_logits.argmax(1).eq(target).sum().item()
            adv_proxy = pgd_inf(proxy_model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            self_adv_correct += proxy_model(adv_proxy).argmax(1).eq(target).sum().item()
            transfer_adv_correct += target_model(adv_proxy).argmax(1).eq(target).sum().item()
    return correct/total, self_adv_correct/total, transfer_adv_correct/total


if __name__ == '__main__':
    device = 'cuda:2'
    args = get_args()

    test_loader = get_test_loader(args.batch_size)
    train_loader = get_train_loader(args.batch_size)

    target_model = eval(args.target_model_name)(num_classes=args.num_classes).to(device)
    target_ckpt = torch.load(args.target_ckpt_path, map_location=device)
    target_model.load_state_dict(target_ckpt)
    target_model.eval()

    proxy_model = eval(args.proxy_model_name)(num_classes=args.num_classes).to(device)
    proxy_optimizer = SGD(proxy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    proxy_scheduler = MultiStepLR(optimizer=proxy_optimizer, milestones=args.milestones, gamma=args.gamma)
    path = os.path.join('logs', args.target_model_name+'_'+args.proxy_model_name)
    path = os.path.join(path, 'adv_'+str(args.adv_distill))
    os.makedirs(path, exist_ok=True)
    logger = SummaryWriter(logdir=path)
    logger.add_text('args', str(args))
    distill()

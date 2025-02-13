"""
Train script for ResNet18 model on MNIST dataset.

Usage:
    python train.py --dataset mnist --data_mean [0.5] --data_std [0.5] --img_size 28 --epochs 4

Reference: https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py
"""

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from stable_signature.models.resnet.resnet18 import ResNet18, BasicBlock
from stable_signature.utils import tuple_inst


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    parser.add_argument('--dataset', required=True,
                        help='cifar10 | mnist')
    parser.add_argument('--dataroot', default=os.path.join(project_root, 'data'),
                        help='path to dataset')
    parser.add_argument('--data_mean', type=tuple_inst(float), default=None)
    parser.add_argument('--data_std', type=tuple_inst(float), default=None)
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--img_channels', type=int, default=1,
                        help='Number of image channels.')
    parser.add_argument('--img_size', type=int, default=28,
                        help='the height / width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default=0.001')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--net', default='',
                        help='path to net (to continue training)')
    parser.add_argument('--outf', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed')

    params = parser.parse_args()
    print(params)
    return params


def main():
    params = parse_args()
    device = torch.device(params.device)
    os.makedirs(params.outf, exist_ok=True)

    if params.manual_seed is None:
        params.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", params.manual_seed)
    random.seed(params.manual_seed)
    torch.manual_seed(params.manual_seed)
    if device.type.startswith('cuda'):
        torch.cuda.manual_seed(params.manual_seed)

    # dataset & data loader
    dataset_cls = getattr(datasets, params.dataset.upper())
    train_dataset = dataset_cls(
        root=params.dataroot, download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(params.data_mean, params.data_std),
        ])
    )
    test_dataset = dataset_cls(
        root=params.dataroot, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(params.data_mean, params.data_std),
        ])
    )
    assert train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=int(params.workers))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=int(params.workers))

    # model
    model = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2],
                     img_channels=params.img_channels,
                     num_classes=params.num_classes).to(device)
    
    if params.net != '':
        model.load_state_dict(torch.load(params.net))

    # loss
    criterion = nn.CrossEntropyLoss()
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # train loop
    for epoch in range(params.epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % (
                    epoch + 1, params.epochs, batch_idx, len(train_loader), loss))

        model.eval()
        print('Epoch: %03d/%03d | Test: %.3f%%' % (
            epoch + 1, params.epochs, eval_loop(model, test_loader, device=device)))

        # do checkpointing
        torch.save(model.state_dict(), f'{params.outf}/resnet18_{epoch:03d}.pth')


@torch.no_grad()
def eval_loop(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        probs = model(features).softmax(1)
        predicted_labels = probs.argmax(1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


if __name__ == '__main__':
    main()

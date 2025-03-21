"""
Train script for DCGAN model on CIFAR10 dataset.

Usage:
    python train.py --dataset cifar10 --dataroot data --image_size 32 --cuda --outf outputs --epochs 100

Reference: https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py
"""

import argparse
import os
import random
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from stable_signature.models.dcgan.dcgan_cifar10 import Generator, Discriminator


def init_weights(m: nn.Module) -> None:
    r"""Custom weights initialization for netG and netD"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    parser.add_argument('--dataset', required=True,
                        help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default=os.path.join(project_root, 'data'),
                        help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--img_size', type=int, default=32,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=64,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--use_dragan', action='store_true', default=False)
    parser.add_argument('--dragan_lambda', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--netG', default='',
                        help='path to netG (to continue training)')
    parser.add_argument('--netD', default='',
                        help='path to netD (to continue training)')
    parser.add_argument('--outf', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--ckpt_freq', type=int, default=5,
                        help='checkpointing frequency')
    parser.add_argument('--manual_seed', type=int,
                        help='manual seed')

    params = parser.parse_args()
    print(params)
    return params


def main():
    params = parse_args()
    device = torch.device(params.device)

    try:
        os.makedirs(params.outf)
    except OSError:
        pass

    if params.manual_seed is not None:
        print('Random Seed: ', params.manual_seed)
        random.seed(params.manual_seed)
        torch.manual_seed(params.manual_seed)
        if device.type.startswith('cuda'):
            torch.cuda.manual_seed(params.manual_seed)

    dataset = datasets.CIFAR10(
        root=params.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Resize(params.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params.batch_size, shuffle=True, num_workers=int(params.workers))

    nz = int(params.nz)

    netG = Generator().to(device)
    netG.apply(init_weights)
    if params.netG != '':
        netG.load_state_dict(torch.load(params.netG))
    print(netG)

    netD = Discriminator().to(device)
    netD.apply(init_weights)
    if params.netD != '':
        netD.load_state_dict(torch.load(params.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(params.batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimD = optim.Adam(netD.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

    for epoch in range(params.epochs):
        for i, (X, _) in enumerate(dataloader):
            batch_size = X.size(0)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            X_real = X.to(device)
            label = torch.full((batch_size,), real_label, device=device).float()

            output = netD(X_real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            # gradient penalty
            if params.use_dragan:
                alpha = torch.rand(batch_size, 1).expand(X.size())
                X_hat = torch.tensor(
                    alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * torch.rand(X.size())),
                    requires_grad=True)
                output = netD(X_hat)
                gradients = autograd.grad(
                    outputs=output, inputs=X_hat, grad_outputs=torch.ones_like(output),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = params.dragan_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                gradient_penalty.backward()
            else:
                gradient_penalty = 0

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake + gradient_penalty
            optimD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimG.step()

            print(f'[{epoch + 1}/{params.epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            if i % 100 == 0:
                save_image(X_real,
                           f'{params.outf}/real_samples.png',
                           normalize=True)
                fake = netG(fixed_noise)
                save_image(fake.detach(),
                           f'{params.outf}/fake_samples_{epoch:03d}.png',
                           normalize=True)

        if (epoch + 1) % params.ckpt_freq == 0:
            torch.save(netG.state_dict(), f'{params.outf}/netG_{epoch:03d}.pth')
            torch.save(netD.state_dict(), f'{params.outf}/netD_{epoch:03d}.pth')


if __name__ == '__main__':
    main()

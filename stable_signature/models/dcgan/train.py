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

from stable_signature.models.dcgan.dcgan import Generator, Discriminator


def init_weights(m: nn.Module) -> None:
    r"""Custom weights initialization for generator and discriminator"""
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
    parser.add_argument('--nc', type=int, default=3,
                        help='size of the latent z vector')
    parser.add_argument('--nz', type=int, default=64,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--use_wgan_div', action='store_true', default=False)
    parser.add_argument('--use_dragan', action='store_true', default=False)
    parser.add_argument('--dragan_lambda', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                        help='learning rate, default=0.0002')
    parser.add_argument('--lr_g', type=float, default=0.001,
                        help='generator learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--generator', default='',
                        help='path to generator (to continue training)')
    parser.add_argument('--discriminator', default='',
                        help='path to discriminator (to continue training)')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of training steps for discriminator per iter')
    parser.add_argument('--outf', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--ckpt_freq', type=int, default=50,
                        help='checkpointing frequency')
    parser.add_argument('--manual_seed', type=int, default=None,
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
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if params.nc == 3 else
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params.batch_size, shuffle=True, num_workers=int(params.workers))

    nz = int(params.nz)

    generator = Generator(params.nc, nz=params.nz, ngf=params.ngf).to(device)
    generator.apply(init_weights)
    if params.generator != '':
        generator.load_state_dict(torch.load(params.generator))
    print(generator)

    discriminator = Discriminator(params.nc, ndf=params.ndf).to(device)
    discriminator.apply(init_weights)
    if params.discriminator != '':
        discriminator.load_state_dict(torch.load(params.discriminator))
    print(discriminator)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(params.batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optim_d = optim.Adam(discriminator.parameters(), lr=params.lr_d, betas=(params.beta1, 0.999))
    optim_g = optim.Adam(generator.parameters(), lr=params.lr_g, betas=(params.beta1, 0.999))

    for epoch in range(1, params.epochs + 1):
        for i, (X, _) in enumerate(dataloader):
            X_real = X.to(device)
            if params.use_wgan_div:
                X_real.requires_grad_(True)
            batch_size = X_real.size(0)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            label = torch.full((batch_size,), real_label, device=device).float()

            real_validity = discriminator(X_real)

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            X_fake = generator(noise).detach()
            if params.use_wgan_div:
                X_fake.requires_grad_(True)
            label.fill_(fake_label)
            fake_validity = discriminator(X_fake)

            if params.use_wgan_div:
                # Compute W-div gradient penalty
                real_grad = autograd.grad(
                    real_validity, X_real,
                    grad_outputs=torch.ones_like(real_validity),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.norm(2, dim=1)

                fake_grad = autograd.grad(
                    fake_validity, X_fake, grad_outputs=torch.ones_like(fake_validity),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad_norm = fake_grad.norm(2, dim=1)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm)
                loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
            else:
                loss_d_real = criterion(real_validity, label)
                loss_d_fake = criterion(fake_validity, label)
                loss_d = loss_d_real + loss_d_fake

            # gradient penalty
            if params.use_dragan:
                alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(X_real)
                X_hat = torch.tensor(
                    alpha * X_real.data +
                    (1 - alpha) * (X_real.data + 0.5 * X_real.data.std() * torch.rand_like(X_real)),
                    requires_grad=True)
                perm_validity = discriminator(X_hat)
                gradients = autograd.grad(
                    perm_validity, X_hat, grad_outputs=torch.ones_like(perm_validity),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = params.dragan_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                loss_d += gradient_penalty

            loss_d.backward()
            optim_d.step()

            if i % params.n_critic == 0:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                X_fake = generator(noise)
                fake_validity = discriminator(X_fake)
                label.fill_(real_label)  # fake labels are real for generator cost
                loss_g = criterion(fake_validity, label)
                loss_g.backward()
                optim_g.step()
                print(f'[{epoch}/{params.epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}')

            if i == 0:
                save_image(X_real.detach(),
                           f'{params.outf}/real_samples.png',
                           normalize=True)
                X_fake = generator(fixed_noise)
                save_image(X_fake.detach(),
                           f'{params.outf}/fake_samples_{epoch:03d}.png',
                           normalize=True)

        if epoch % params.ckpt_freq == 0 or epoch == params.epochs:
            torch.save(generator.state_dict(), f'{params.outf}/generator_{epoch:03d}.pth')
            torch.save(discriminator.state_dict(), f'{params.outf}/discriminator_{epoch:03d}.pth')


if __name__ == '__main__':
    main()

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tv_transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden.ops import attacks as hidden_attacks, metrics as hidden_metrics, transforms as hidden_transforms
from hidden.models.attack_layers import HiddenAttackLayer
from stable_signature import utils
from stable_signature.models import hidden_utils
from stable_signature.models import dcgan


def parse_args(verbose: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(__file__))

    g = parser.add_argument_group('Data parameters')
    g.add_argument('exp', type=str, nargs='?', default=None, help='Experiment name')
    g.add_argument('--dataset', type=str, default=None)
    g.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data'))
    g.add_argument('--train_dir', type=str, default=None)
    g.add_argument('--val_dir', type=str, default=None)
    g.add_argument('--output_dir', type=str, default='outputs',
                   help='Output directory for logs and images (Default: output)')
    g.add_argument('--data_mean', type=utils.tuple_inst(float), default=None)
    g.add_argument('--data_std', type=utils.tuple_inst(float), default=None)

    g = parser.add_argument_group('Model parameters')
    g.add_argument('--generator_ckpt', type=str, required=True,
                   help='Path to the checkpoint file for the Generator')
    g.add_argument('--discriminator_ckpt', type=str, default=None,
                   help='Path to the checkpoint file for the Discriminator')
    g.add_argument('--decoder', type=str, choices=['hidden', 'resnet'], default='hidden',
                   help='Decoder type')
    g.add_argument('--decoder_path', type=str, required=True,
                   help='Path to the hidden decoder for the watermarking model')
    g.add_argument('--clf_ckpt', type=str, default=None,
                   help='Path to the classifier checkpoint for computing distillation loss')
    g.add_argument('--num_bits', type=int, default=16, help='Number of bits in the watermark')
    g.add_argument('--z_dim', type=int, default=100, help='Dimension of the latent vector')
    g.add_argument('--decoder_depth', type=int, default=8,
                   help='Depth of the decoder in the watermarking model')
    g.add_argument('--decoder_channels', type=int, default=64,
                   help='Number of channels in the decoder of the watermarking model')

    g = parser.add_argument_group('Attack layer parameters')
    g.add_argument('--attack_layer', type=str, default='hidden',
                   help='Type of data augmentation to use at marking time. (Default: hidden)')
    g.add_argument('--p_flip', type=float, default=1,
                   help='Probability of the flip attack. (Default: 1)')
    g.add_argument('--p_drop', type=float, default=1,
                   help='Probability of the watermark dropout attack. (Default: 1)')
    g.add_argument('--p_color_jitter', '--p_color_jiggle', type=float, default=1,
                   help='Probability of the color jitter attack. (Default: 1)')
    g.add_argument('--p_crop', type=float, default=0,
                   help='Probability of the crop attack. (Default: 0)')
    g.add_argument('--p_resize', type=float, default=1,
                   help='Probability of the resize attack. (Default: 1)')
    g.add_argument('--p_blur', type=float, default=1,
                   help='Probability of the blur attack. (Default: 1)')
    g.add_argument('--p_rotate', '--p_rot', type=float, default=1,
                   help='Probability of the rotation attack. (Default: 1)')
    g.add_argument('--p_jpeg', type=float, default=1,
                   help='Probability of the diff JPEG attack. (Default: 1)')
    g.add_argument('--p_jpeg2000', type=float, default=0,
                   help='Probability of the diff JPEG2000 attack. (Default: 0)')
    g.add_argument('--p_webp', type=float, default=0,
                   help='Probability of the diff WebP attack. (Default: 0)')

    g = parser.add_argument_group('Training parameters')
    g.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    g.add_argument('--workers', type=int, default=8,
                   help='Number of workers for data loading. (Default: 8)')
    g.add_argument('--img_size', type=int, default=256, help='Resize images to this size')
    g.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')

    g.add_argument('--loss_c', type=str, default='bce',
                   help='Type of loss for the critic loss. Can be bce')
    g.add_argument('--loss_w', type=str, default='bce',
                   help='Type of loss for the watermark loss. Can be mse or bce')
    g.add_argument('--lambda_c', type=float, default=1.0,
                   help='Weight of the image loss in the total loss')
    g.add_argument('--lambda_w', type=float, default=1.0,
                   help='Weight of the watermark loss in the total loss')
    g.add_argument('--lambda_dragan', type=float, default=1.0,
                   help='Weight of the DRAGAN gradient penalty')
    g.add_argument('--optimizer', type=str, default='AdamW,lr=5e-4',
                   help='Optimizer and learning rate for training')
    g.add_argument('--steps', type=int, default=100,
                   help='Number of steps to train the model for')
    g.add_argument('--critic_steps', type=int, default=5,
                   help='Number of steps to train the discriminator')
    g.add_argument('--warmup_steps', type=int, default=20,
                   help='Number of warmup steps for the optimizer')

    g = parser.add_argument_group('Eval parameters')
    g.add_argument('--eval_steps', type=int, default=100,
                   help='Number of steps to evaluate the model for')
    g.add_argument('--eval_freq', type=int, default=200,
                   help='Eval frequency')
    g.add_argument('--eval_seed', type=int, default=1)

    g = parser.add_argument_group('Logging and saving freq. parameters')
    g.add_argument('--log_train_metrics', type=utils.bool_inst, default=True)
    g.add_argument('--log_freq', type=int, default=10, help='Logging frequency (in steps)')
    g.add_argument('--save_img_freq', type=int, default=100,
                   help='Frequency of saving generated images (in steps)')

    g = parser.add_argument_group('Distributed training parameters')
    g.add_argument('--device', type=str, default='cuda:0', help='Device')

    g = parser.add_argument_group('Experiments parameters')
    g.add_argument('--num_keys', type=int, default=1,
                   help='Number of fine-tuned checkpoints to generate')
    g.add_argument('--seed', type=int, default=0)

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)

    if params.attack_layer is not None:
        if params.attack_layer.lower() == 'none':
            params.attack_layer = None

    # Print the arguments
    if verbose:
        print(params)

    params.device = torch.device(params.device) if torch.cuda.is_available() else torch.device('cpu')
    return params


def main():
    params = parse_args()

    # Set seeds for reproducibility
    if params.seed is not None:
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        np.random.seed(params.seed)

    # Normalization
    params.normalize = hidden_attacks.ImageNormalize(mean=params.data_mean, std=params.data_std).to(params.device)
    params.denormalize = hidden_attacks.ImageDenormalize(mean=params.data_mean, std=params.data_std).to(params.device)

    # Data loaders
    train_transform = tv_transforms.Compose([
        tv_transforms.Resize(params.img_size),
        tv_transforms.CenterCrop(params.img_size),
        tv_transforms.ToTensor(),
        hidden_transforms.Normalize(params.data_mean, params.data_std),
    ])
    if params.dataset is not None:
        train_loader = utils.get_dataloader(params.data_dir, dataset=params.dataset, train=True,
                                            transform=train_transform, batch_size=params.batch_size,
                                            num_workers=params.workers, shuffle=True)
    else:
        train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size,
                                            num_workers=params.workers, shuffle=True)

    # Create output dirs
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    params.imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(params.imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building Generator and Discriminator...')
    G0 = dcgan.Generator(params.img_channels, params.z_dim).to(params.device)
    G0.load_state_dict(torch.load(params.generator_ckpt, weights_only=False, map_location=params.device))
    G0.eval()
    D0 = dcgan.Discriminator(params.img_channels).to(params.device)
    D0.load_state_dict(torch.load(params.discriminator_ckpt, weights_only=False, map_location=params.device))
    D0.eval()

    # Loads attack layer
    if params.attack_layer is None:
        attack_layer = None
    elif params.attack_layer == 'hidden':
        attack_layer = HiddenAttackLayer(
            params.img_size,
            p_flip=params.p_flip,
            p_drop=params.p_drop,
            p_color_jitter=params.p_color_jitter if params.img_channels == 3 else 0,
            p_crop=params.p_crop,
            p_resize=params.p_resize,
            p_blur=params.p_blur,
            p_rotate=params.p_rotate,
            p_jpeg=params.p_jpeg,
            p_jpeg2000=params.p_jpeg2000,
            p_webp=params.p_webp,
            mean=params.data_mean, std=params.data_std).to(params.device)
    else:
        raise ValueError('attack_layer not recognized')

    # Loads hidden decoder
    print(f'>>> Building HiDDeN Decoder...')
    msg_decoder = hidden_utils.get_hidden_decoder(params.decoder,
                                                  num_bits=params.num_bits,
                                                  num_blocks=params.decoder_depth,
                                                  channels=params.decoder_channels,
                                                  in_channels=params.img_channels).to(params.device)
    msg_decoder.load_state_dict(hidden_utils.get_hidden_decoder_ckpt(params.decoder_path))
    # TODO: add whitening?
    msg_decoder.eval()

    # Freeze hidden decoder
    for param in [*msg_decoder.parameters()]:
        param.requires_grad = False

    img_transform = nn.Identity()

    # Create losses
    print(f'>>> Creating Losses...')
    if params.loss_w == 'mse':
        message_loss = lambda m_hat, m, temp=10.0: torch.mean((m_hat * temp - (2 * m - 1)) ** 2)  # b k - b k
    elif params.loss_w == 'bce':
        message_loss = lambda m_hat, m, temp=10.0: torch.nn.functional.binary_cross_entropy_with_logits(
            m_hat * temp, m, reduction='mean')
    else:
        raise ValueError(f'Unknown message loss: {params.loss_w}')

    critic_loss = nn.BCEWithLogitsLoss()

    # attacks
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_08': hidden_attacks.CenterCrop(0.8),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'resize_08': hidden_attacks.Resize2(0.8),
        'resize_05': hidden_attacks.Resize2(0.5),
        'rot_25': hidden_attacks.Rotate(25),
        'rot_90': hidden_attacks.Rotate(90),
        'blur': hidden_attacks.GaussianBlur(
            kernel_size=5, sigma=0.5, mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_80': hidden_attacks.JPEGCompress(
            80, mean=params.data_mean, std=params.data_std).to(params.device),
    }

    # Construct metrics
    metrics = {
        'psnr': hidden_metrics.PSNR(mean=params.data_mean, std=params.data_std).to(params.device),
        'ssim': hidden_metrics.SSIM(mean=params.data_mean, std=params.data_std).to(params.device),
    }
    if params.img_size >= 160:
        metrics.update({
            'ms_ssim': hidden_metrics.MS_SSIM(mean=params.data_mean, std=params.data_std).to(params.device),
        })
    if params.img_channels == 3:
        metrics.update({
            'lpips': hidden_metrics.LPIPS(net='alex').to(params.device),
        })

    for ii_key in range(params.num_keys):
        # Creating key
        print(f'\n>>> Creating key with {params.num_bits} bits...')
        key = torch.randint(0, 2, (1, params.num_bits), dtype=torch.float32, device=params.device)
        key_str = ''.join([str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')
        with (Path(params.output_dir) / 'keys.txt').open('a') as f:
            f.write(f'{ii_key:03d}\t{key_str}\n')

        # Copy the Generator and finetune the copy
        G = deepcopy(G0).to(params.device)
        D = deepcopy(D0).to(params.device)
        for param in G.parameters():
            param.requires_grad = True
        optim_params = utils.parse_initializer_params(params.optimizer)
        optim_g = utils.build_optimizer(model_params=G.parameters(), **optim_params)
        optim_d = utils.build_optimizer(model_params=D.parameters(), **{**optim_params, 'lr': optim_params['lr'] / 10})

        # Training loop
        print(f'>>> Training...')
        start_iter = 0
        while start_iter < params.steps:
            train_stats = train(train_loader, optim_g, optim_d, message_loss, critic_loss,
                                G, D, attack_layer, msg_decoder, img_transform, key,
                                metrics, start_iter + 1, params)
            val_stats = val(G, msg_decoder, img_transform, key, eval_attacks,
                            metrics, start_iter + 1, params)
            start_iter = min(start_iter + params.eval_freq, params.steps)
            log_stats = {'it': start_iter,
                         **{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         }
            save_dict = {
                'generator': G.state_dict(),
                'params': params,
            }

            # Save checkpoint
            torch.save(save_dict, os.path.join(params.output_dir, f'checkpoint_{ii_key:03d}.pth'))
            with (Path(params.output_dir) / f'log_{ii_key:03d}.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
            print('\n')


def itemize(tensor):
    if torch.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return tensor.item()
    return tensor


def train(train_loader, optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer, message_loss, critic_loss,
          G: nn.Module, D: nn.Module, attack_layer: nn.Module, msg_decoder: nn.Module, img_transform,
          key: torch.Tensor, metrics: Dict, start_iter: int, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger()
    G.train()

    base_lr = optim_g.param_groups[0]['lr']
    m = key.repeat(params.batch_size, 1)
    ori_msgs = torch.sign(m) > 0
    for it in metric_logger.log_every(range(start_iter, min(start_iter + params.eval_freq, params.steps) + 1),
                                      params.log_freq, header):
        # --------------------
        # Update discriminator
        # --------------------
        train_loader_iter = iter(train_loader)
        for jt in range(params.critic_steps):
            optim_d.zero_grad()

            # train with real
            x_real, y_real = next(train_loader_iter)
            x_real = x_real.to(params.device)
            real_validity = D(x_real)
            loss_d_real = critic_loss(real_validity, torch.ones_like(real_validity))
            loss_d_real.backward()

            # train with fake
            z = torch.randn(params.batch_size, params.z_dim, 1, 1, device=params.device)  # b z 1 1
            x_fake = G(z).detach()
            fake_validity = D(x_fake)
            loss_d_fake = critic_loss(fake_validity, torch.zeros_like(fake_validity))
            loss_d_fake.backward()

            # dragan
            if params.lambda_dragan:
                alpha = torch.rand(params.batch_size, 1, 1, 1, device=params.device).expand_as(x_real)
                x_hat = alpha * x_real + (1 - alpha) * (x_real + 0.5 * x_real.std() * torch.rand_like(x_real))
                x_hat = x_hat.clone().detach().requires_grad_(True)
                perm_validity = D(x_hat)
                gradients = torch.autograd.grad(
                    perm_validity, x_hat, grad_outputs=torch.ones_like(perm_validity),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = params.lambda_dragan * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                gradient_penalty.backward()

            optim_d.step()

        # --------------------
        # Update generator
        # --------------------
        utils.adjust_learning_rate(optim_g, it, params.steps, params.warmup_steps, base_lr)
        # random latent vector
        z = torch.randn(params.batch_size, params.z_dim, 1, 1, device=params.device)  # b z 1 1
        # decode latents with original and fine-tuned decoder
        x_w = G(z)  # b z 1 1 -> b c h w
        validity_w = D(x_w)  # b z h w -> b 1

        # simulated attacks
        x_r = attack_layer(x_w, None) if attack_layer is not None else x_w
        # extract watermark
        m_hat = msg_decoder(img_transform(x_r))  # b c h w -> b k

        # compute loss
        loss_w = message_loss(m_hat, m)
        loss_c = critic_loss(validity_w, torch.ones_like(validity_w))
        loss = params.lambda_w * loss_w + params.lambda_c * loss_c

        # optim step
        loss.backward()
        optim_g.step()
        optim_g.zero_grad()

        # log stats
        decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
        bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
        word_accs = bit_accs == 1  # b
        log_stats = {
            'lr': optim_g.param_groups[0]['lr'],
            'loss': itemize(loss),
            'loss_w': itemize(loss_w),
            'loss_c': itemize(loss_c),
            'bit_acc': torch.mean(bit_accs).item(),
            'word_acc': torch.mean(word_accs.float()).item(),
        }
        # if params.log_train_metrics and it % 100 == 0:
        #     log_stats.update({
        #         **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()}
        #     })

        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        # save images during training
        if it % params.save_img_freq == 0:
            save_image(torch.clamp(params.denormalize(x_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_train_xw.png'), nrow=8)

    print(f'✔️ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val(G: nn.Module, msg_decoder: nn.Module, img_transform,
        key: torch.Tensor, eval_attacks: Dict, metrics: Dict, start_iter: int, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger()
    G.eval()

    m = key.repeat(params.batch_size, 1)
    ori_msgs = torch.sign(m) > 0
    # assuring same latent vectors generated
    generator = torch.Generator(device=params.device).manual_seed(params.eval_seed)
    for it in metric_logger.log_every(range(1, params.eval_steps + 1), params.log_freq, header):
        # random latent vector
        z = torch.randn(params.batch_size, params.z_dim, 1, 1, device=params.device, generator=generator)  # b z 1 1
        # decode latents with original and fine-tuned decoder
        x_w = G(z)  # b z 1 1 -> b c h w

        log_stats = {}
        # log_stats = {
        #     **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()},
        # }
        for name, attack in eval_attacks.items():
            x_r = attack(img_transform(x_w))
            m_hat = msg_decoder(x_r)  # b c h w -> b k
            decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
            bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if it == 1:
            save_image(torch.clamp(params.denormalize(x_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{start_iter:05d}_val_xw.png'), nrow=8)

    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

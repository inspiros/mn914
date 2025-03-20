# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
torchrun --nproc_per_node=2 main.py \
    --local_rank 0 \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --scaling_w 0.5 \
    --batch_size 16 --eval_freq 10 \
    --attenuation jnd \
    --epochs 100 --optimizer 'AdamW,lr=1e-4'
    
Args Inventory:
    --dist False \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --batch_size 128 --workers 8 \
    --attenuation jnd \
    --num_bits 64 --redundancy 16 \
    --encoder vit --encoder_depth 6 --encoder_channels 384 --use_tanh True \
    --encoder vit --encoder_depth 12 --encoder_channels 384 --use_tanh True \
    --attenuation jnd --batch_size 16 --eval_freq 10 --local_rank 0 \
    --p_crop 0 --p_rot 0 --p_color_jitter 0 --p_blur 0 --p_jpeg 0 --p_resize 0 \

"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image

# add hidden path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden import models, utils
from hidden.models import attenuations, attack_layers
from hidden.ops import attacks as hidden_attacks, metrics as hidden_metrics, transforms as hidden_transforms

from stable_signature.loss.loss_provider import LossProvider


def parse_args(verbose: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(__file__))

    g = parser.add_argument_group('Experiments parameters')
    g.add_argument('exp', type=str, nargs='?', default=None, help='Experiment name')
    g.add_argument('--dataset', type=str, default=None)
    g.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data'))
    g.add_argument('--train_dir', type=str, default=None)
    g.add_argument('--val_dir', type=str, default=None)
    g.add_argument('--output_dir', type=str, default='outputs',
                   help='Output directory for logs and images (Default: outputs)')
    g.add_argument('--data_mean', type=utils.tuple_inst(float), default=None)
    g.add_argument('--data_std', type=utils.tuple_inst(float), default=None)
    g.add_argument('--eval_only', action='store_true', default=False)

    g = parser.add_argument_group('Marking parameters')
    g.add_argument('--num_bits', type=int, default=32,
                   help='Number of bits of the watermark (Default: 32)')
    g.add_argument('--img_size', type=int, default=28, help='Image size')
    g.add_argument('--img_channels', type=int, default=None, help='Number of image channels.')

    g = parser.add_argument_group('Encoder parameters')
    g.add_argument('--encoder', type=str, default='hidden',
                   help='Encoder type (Default: hidden)')
    g.add_argument('--encoder_depth', default=4, type=int,
                   help='Number of blocks in the encoder.')
    g.add_argument('--encoder_channels', default=64, type=int,
                   help='Number of channels in the encoder.')
    g.add_argument('--use_tanh', type=utils.bool_inst, default=True,
                   help='Use tanh scaling. (Default: True)')
    g.add_argument('--generate_delta', type=utils.bool_inst, default=True,
                   help='Generate permutation delta instead of watermarked image directly (Default: True).')

    g = parser.add_argument_group('Decoder parameters')
    g.add_argument('--decoder', type=str, default='hidden',
                   help='Decoder type (Default: hidden)')
    g.add_argument('--decoder_depth', type=int, default=8,
                   help='Number of blocks in the decoder (Default: 4)')
    g.add_argument('--decoder_channels', type=int, default=64,
                   help='Number of blocks in the decoder (Default: 4)')

    g = parser.add_argument_group('Training parameters')
    g.add_argument('--eval_freq', default=10, type=int)
    g.add_argument('--saveckpt_freq', default=100, type=int)
    g.add_argument('--saveimg_freq', default=10, type=int)
    g.add_argument('--resume_from', default=None, type=str,
                   help='Checkpoint path to resume from.')
    g.add_argument('--scaling_w', type=float, default=1.0,
                   help='Scaling of the watermark signal. (Default: 1.0)')
    g.add_argument('--scaling_i', type=float, default=1.0,
                   help='Scaling of the original image. (Default: 1.0)')

    g = parser.add_argument_group('Optimization parameters')
    g.add_argument('--epochs', type=int, default=100,
                   help='Number of epochs for optimization. (Default: 100)')
    g.add_argument('--reconstruct_pretrain_epochs', type=int, default=0,
                   help='Number of epochs for image loss-only pretraining. (Default: 0)')
    g.add_argument('--encoder_only_epochs', type=int, default=0,
                   help='Number of epochs for pretraining encoder only. (Default: 0)')
    g.add_argument('--optimizer', type=str, default='Adam',
                   help='Optimizer to use. (Default: Adam)')
    g.add_argument('--scheduler', type=utils.nullable(str), default=None,
                   help='Scheduler to use. (Default: None)')

    g.add_argument('--loss_w', type=str, default='bce',
                   help='Message Loss: bce | mse | cossim')
    g.add_argument('--loss_i', type=str, default='mse',
                   help='Image Loss: mse (Default: mse)')
    g.add_argument('--loss_p', type=utils.nullable(str), default=None,
                   help='Perceptual Loss: lpips | watson-dft | watson-dct | watson-vgg (Default: none)')
    g.add_argument('--loss_p_dir', type=str, default=os.path.join(project_root, 'ckpts/loss'),
                   help='Pretrained weights dir for perceptual loss.')

    g.add_argument('--lambda_w', type=float, default=1.0,
                   help='Weight of the watermark loss. (Default: 1.0)')
    g.add_argument('--lambda_i', type=float, default=0.0,
                   help='Weight of the image loss. (Default: 0.0)')
    g.add_argument('--lambda_p', type=float, default=1.0,
                   help='Weight of the perceptual loss. (Default: 1.0)')

    g = parser.add_argument_group('Loader parameters')
    g.add_argument('--batch_size', type=int, default=64, help='Batch size. (Default: 64)')
    g.add_argument('--workers', type=int, default=8,
                   help='Number of workers for data loading. (Default: 8)')

    g = parser.add_argument_group('Attenuation parameters')
    g.add_argument('--attenuation', type=utils.nullable(str), default=None,
                   help='Attenuation type. (Default: None)')
    g.add_argument('--scale_channels', type=utils.bool_inst, default=False,
                   help='Use channel scaling. (Default: False)')

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

    g = parser.add_argument_group('Distributed training parameters')
    g.add_argument('--local_rank', default=-1, type=int)
    g.add_argument('--master_port', default=-1, type=int)
    g.add_argument('--dist', type=utils.bool_inst, default=False,
                   help='Enabling distributed training')
    g.add_argument('--device', type=str, default='cuda:0', help='Device')

    g = parser.add_argument_group('Misc')
    g.add_argument('--seed', default=None, type=utils.nullable(int), help='Random seed')
    g.add_argument('--eval_seed', default=0, type=int, help='Random seed for evaluation')

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)
    if (params.data_mean is None) ^ (params.data_std is None):
        raise ValueError('Data mean and std are both required.')

    if params.dist:
        params.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print the arguments
    if verbose:
        print('git:', utils.get_sha())
        print(json.dumps(vars(params)))

    return params


def main():
    params = parse_args()

    # Distributed mode
    if params.dist:
        utils.init_distributed_mode(params)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    # Set seeds for reproducibility
    if params.seed is not None:
        seed = params.seed + utils.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Create output dirs
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    params.imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(params.imgs_dir, exist_ok=True)
    with open(os.path.join(params.output_dir, 'params.json'), 'w') as f:
        json.dump(vars(params), f)

    # Normalization
    params.normalize = hidden_attacks.ImageNormalize(params.data_mean, params.data_std).to(params.device)
    params.denormalize = hidden_attacks.ImageDenormalize(params.data_mean, params.data_std).to(params.device)

    # Data loaders
    train_transform = tv_transforms.Compose([
        tv_transforms.Resize(params.img_size),
        tv_transforms.CenterCrop(params.img_size),
        tv_transforms.RandomResizedCrop(params.img_size, scale=(0.5, 1.0)),
        tv_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.ToTensor(),
        hidden_transforms.Normalize(params.data_mean, params.data_std),
    ])
    val_transform = tv_transforms.Compose([
        tv_transforms.Resize(params.img_size),
        tv_transforms.CenterCrop(params.img_size),
        tv_transforms.ToTensor(),
        hidden_transforms.Normalize(params.data_mean, params.data_std),
    ])
    if params.dataset is not None:
        train_loader = utils.get_dataloader(params.data_dir, dataset=params.dataset, train=True,
                                            transform=train_transform, batch_size=params.batch_size,
                                            num_workers=params.workers, shuffle=True)
        val_loader = utils.get_dataloader(params.data_dir, dataset=params.dataset, train=False,
                                          transform=val_transform, batch_size=params.batch_size,
                                          num_workers=params.workers, shuffle=False)
    else:
        train_loader = utils.get_dataloader(params.train_dir, transform=train_transform, batch_size=params.batch_size,
                                            num_workers=params.workers, shuffle=True)
        val_loader = utils.get_dataloader(params.val_dir, transform=val_transform, batch_size=params.batch_size,
                                          num_workers=params.workers, shuffle=False)

    # Input shape
    if params.img_channels is None:
        if params.dataset is not None:
            _shape_infer_loader = utils.get_dataloader(params.data_dir, dataset=params.dataset, train=False,
                                                       transform=tv_transforms.ToTensor(), batch_size=1,
                                                       num_workers=1, shuffle=False)
        else:
            _shape_infer_loader = utils.get_dataloader(params.train_dir, transform=tv_transforms.ToTensor(),
                                                       batch_size=1, num_workers=1, shuffle=False)
        params.img_channels = _shape_infer_loader.dataset[0][0].size(-3)
        del _shape_infer_loader

    # Build encoder
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits,
                                       channels=params.encoder_channels, in_channels=params.img_channels,
                                       last_tanh=params.use_tanh)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits,
                                       channels=params.encoder_channels, in_channels=params.img_channels,
                                       last_tanh=params.use_tanh)
    elif params.encoder == 'vit':
        encoder = models.VitEncoder(
            img_size=params.img_size, patch_size=16, init_values=None,
            embed_dim=params.encoder_channels, depth=params.encoder_depth,
            num_bits=params.num_bits, in_channels=params.img_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'ae':
        encoder = models.AEHidingNetwork(
            num_bits=params.num_bits, in_channels=params.img_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'unet':
        encoder = models.UNetHidingNetwork(
            num_bits=params.num_bits, in_channels=params.img_channels, last_tanh=params.use_tanh)
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n', encoder)
    print('total parameters:', sum(p.numel() for p in encoder.parameters()))

    # Build decoder
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth,
                                       num_bits=params.num_bits,
                                       channels=params.decoder_channels,
                                       in_channels=params.img_channels)
    elif params.decoder == 'resnet':
        decoder = models.resnet50_decoder(num_bits=params.num_bits,
                                          img_channels=params.img_channels,
                                          low_resolution=True)
    else:
        raise ValueError('Unknown decoder type')
    print('\ndecoder: \n', decoder)
    print('total parameters:', sum(p.numel() for p in decoder.parameters()))

    # Construct attenuation
    if params.attenuation == 'jnd':
        attenuation = attenuations.JND(
            preprocess=hidden_attacks.ImageDenormalize(params.data_mean, params.data_std)).to(params.device)
    else:
        attenuation = None

    # Construct data augmentation seen at train time
    if params.attack_layer == 'hidden':
        attack_layer = attack_layers.HiddenAttackLayer(
            params.img_size,
            p_flip=params.p_flip,
            p_drop=params.p_drop,
            p_color_jitter=params.p_color_jitter,
            p_crop=params.p_crop,
            p_resize=params.p_resize,
            p_blur=params.p_blur,
            p_rotate=params.p_rotate,
            p_jpeg=params.p_jpeg,
            p_jpeg2000=params.p_jpeg2000,
            p_webp=params.p_webp,
            mean=params.data_mean, std=params.data_std).to(params.device)
    elif params.attack_layer == 'none':
        attack_layer = None
    else:
        raise ValueError('Unknown attack layer')
    print('Attack Layer:', attack_layer)

    print(f'Losses: {params.loss_w} and {params.loss_i}')
    if params.loss_w == 'mse':
        message_loss = lambda m_hat, m: torch.mean((m_hat - (2 * m - 1)) ** 2)  # b k - b k
    elif params.loss_w == 'bce':
        message_loss = lambda m_hat, m: torch.nn.functional.binary_cross_entropy_with_logits(m_hat, m, reduction='mean')
    else:
        raise ValueError(f'Unknown message loss: {params.loss_w}')

    if params.loss_i == 'mse':
        image_loss = lambda x_w, x0: torch.mean((x_w - x0) ** 2)
    else:
        raise ValueError(f'Unknown image loss: {params.loss_i}')

    if params.loss_p is not None:
        provider = LossProvider(params.loss_p_dir)
        colorspace = 'LA' if params.img_channels == 1 else 'RGB'
        if params.loss_p == 'watson-dft':
            loss_func = provider.get_loss_function(
                'Watson-DFT', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
            perceptual_loss = lambda x_w, x0: loss_func((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
        elif params.loss_p == 'watson-vgg':
            loss_func = provider.get_loss_function(
                'Watson-VGG', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
            perceptual_loss = lambda x_w, x0: loss_func((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
        elif params.loss_p == 'ssim':
            loss_func = provider.get_loss_function(
                'SSIM', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
            perceptual_loss = lambda x_w, x0: loss_func((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
        else:
            raise ValueError(f'Unknown perceptual loss: {params.loss_p}')
    else:
        perceptual_loss = None

    # attacks
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_03': hidden_attacks.CenterCrop(0.3),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'resize_03': hidden_attacks.Resize(0.3),
        'resize_05': hidden_attacks.Resize(0.5),
        'rot_25': hidden_attacks.Rotate(25),
        'rot_90': hidden_attacks.Rotate(90),
        # 'brightness_2': hidden_attacks.AdjustBrightness(2, mean=params.data_mean, std=params.data_std).to(params.device),
        'blur': hidden_attacks.GaussianBlur(kernel_size=5, sigma=0.5,
                                            mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_90': hidden_attacks.JPEGCompress(90,
                                               mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_80': hidden_attacks.JPEGCompress(80,
                                               mean=params.data_mean, std=params.data_std).to(params.device),
    }
    eval_attacks = {k: attack_layers.wrap_attack(v, False) for k, v in eval_attacks.items()}

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

    # Create encoder/decoder
    encoder_decoder = models.EncoderDecoder(encoder=encoder,
                                            attenuation=attenuation,
                                            attack_layer=attack_layer,
                                            decoder=decoder,
                                            generate_delta=params.generate_delta,
                                            scale_channels=params.scale_channels,
                                            scaling_i=params.scaling_i,
                                            scaling_w=params.scaling_w,
                                            num_bits=params.num_bits,
                                            std=params.data_std if params.scale_channels else None)
    encoder_decoder = encoder_decoder.to(params.device)

    # Distributed training
    if params.dist:
        encoder_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder_decoder)
        encoder_decoder = nn.parallel.DistributedDataParallel(encoder_decoder, device_ids=[params.local_rank])

    # Build optimizer and scheduler
    optim_params = utils.parse_initializer_params(params.optimizer)
    lr_mult = params.batch_size * utils.get_world_size() / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    to_optim = [*encoder.parameters(), *decoder.parameters()]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_initializer_params(
        params.scheduler)) if params.scheduler is not None else None
    print('optimizer:', optimizer)
    print('scheduler:', scheduler)

    # optionally resume training
    to_restore = {'epoch': 1}
    if os.path.isfile(os.path.join(params.output_dir, 'checkpoint.pth')):
        utils.restart_from_checkpoint(
            os.path.join(params.output_dir, 'checkpoint.pth'),
            run_variables=to_restore,
            encoder_decoder=encoder_decoder,
            optimizer=optimizer,
        )
    elif params.resume_from is not None:
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
        if params.encoder_only_epochs:
            decoder.requires_grad_(False)
    start_epoch = to_restore['epoch']
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    if params.eval_only:
        params.output_dir = os.path.join(params.output_dir, 'eval')
        os.makedirs(params.output_dir, exist_ok=True)
        print('evaluating...')
        val_stats = eval_one_epoch(encoder_decoder, train_loader,
                                   message_loss, image_loss, perceptual_loss,
                                   eval_attacks, metrics, start_epoch, params)
        log_stats = {'epoch': start_epoch, **{f'val_{k}': v for k, v in val_stats.items()}}
        if utils.is_main_process():
            with (Path(params.output_dir) / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
        exit()

    # train loop
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs + 1):
        if params.dist:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        if params.resume_from is not None and epoch >= params.encoder_only_epochs:
            decoder.requires_grad_(True)

        train_stats = train_one_epoch(encoder_decoder, train_loader, optimizer,
                                      message_loss, image_loss, perceptual_loss,
                                      scheduler, metrics, epoch, params)
        log_stats = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()}}

        if epoch % params.eval_freq == 0 or epoch == params.epochs:
            val_stats = eval_one_epoch(encoder_decoder, val_loader,
                                       message_loss, image_loss, perceptual_loss,
                                       eval_attacks, metrics, epoch, params)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}

        if utils.is_main_process():
            with (Path(params.output_dir) / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        utils.save_on_master(save_dict, os.path.join(params.output_dir, 'checkpoint.pth'))
        if (params.saveckpt_freq and epoch % params.saveckpt_freq == 0) or epoch == params.epochs:
            utils.save_on_master(save_dict, os.path.join(params.output_dir, f'checkpoint{epoch:03d}.pth'))

    total_time = time.time() - start_time
    print(f'Training time {datetime.timedelta(seconds=int(total_time))}')


def itemize(tensor):
    if torch.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return tensor.item()
    return tensor


# noinspection DuplicatedCode
def train_one_epoch(encoder_decoder: models.EncoderDecoder, loader, optimizer,
                    message_loss, image_loss, perceptual_loss,
                    scheduler, metrics, epoch, params):
    r"""
    One epoch of training.
    """
    if params.scheduler is not None:
        scheduler.step(epoch)
    header = f'[Epoch {epoch}/{params.epochs}]'
    encoder_decoder.train()
    metric_logger = utils.MetricLogger()

    for it, (x0, _) in enumerate(metric_logger.log_every(loader, 10, f'+ {header}')):
        x0 = x0.to(params.device, non_blocking=True)  # b c h w

        m = torch.randint(0, 2, (x0.size(0), params.num_bits), dtype=torch.float32, device=params.device)  # b k [0 1]
        m_normalized = 2 * m - 1  # b k [-1 1]

        m_hat, (x_w, x_r) = encoder_decoder(x0, m_normalized)

        loss_w = message_loss(m_hat, m) if epoch >= params.reconstruct_pretrain_epochs else 0
        loss_i = image_loss(x_w, x0)  # b c h w -> 1
        loss_p = perceptual_loss(x_w, x0) if perceptual_loss is not None else 0
        loss = params.lambda_w * loss_w + params.lambda_i * loss_i + params.lambda_p * loss_p

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        ori_msgs = torch.sign(m) > 0
        decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
        bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
        word_accs = bit_accs == 1  # b
        log_stats = {
            'lr': optimizer.param_groups[0]['lr'],
            'loss': itemize(loss),
            'loss_w': itemize(loss_w),
            'loss_i': itemize(loss_i),
            'loss_p': itemize(loss_p),
            'bit_acc': torch.mean(bit_accs).item(),
            'word_acc': torch.mean(word_accs.type(torch.float)).item(),
        }
        # if epoch % params.eval_freq == 0 or epoch == params.epochs:
        log_stats.update({
            **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()}
        })

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if epoch % params.saveimg_freq == 0 and it == 0 and utils.is_main_process():
            save_image(params.denormalize(x0),
                       os.path.join(params.imgs_dir, f'{epoch:03d}_train_x0.png'), nrow=8)
            save_image(params.denormalize(x_w),
                       os.path.join(params.imgs_dir, f'{epoch:03d}_train_xw.png'), nrow=8)
            save_image(params.denormalize(x_r),
                       os.path.join(params.imgs_dir, f'{epoch:03d}_train_xr.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print(f'✔️ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# noinspection DuplicatedCode
@torch.no_grad()
def eval_one_epoch(encoder_decoder: models.EncoderDecoder, loader,
                   message_loss, image_loss, perceptual_loss,
                   eval_attacks, metrics, epoch, params):
    r"""
    One epoch of eval.
    """
    header = f'[Epoch {epoch}/{params.epochs}]'
    encoder_decoder.eval()
    metric_logger = utils.MetricLogger()

    # assuring same keys generated
    generator = torch.Generator(device=params.device).manual_seed(params.eval_seed)
    for it, (x0, _) in enumerate(metric_logger.log_every(loader, 10, f'- {header}')):
        x0 = x0.to(params.device, non_blocking=True)  # b c h w

        m = torch.randint(0, 2, (x0.size(0), params.num_bits), dtype=torch.float32, device=params.device,
                          generator=generator)  # b k [0 1]
        m_normalized = 2 * m - 1  # b k [-1 1]

        m_hat, (x_w, x_r) = encoder_decoder(x0, m_normalized)

        loss_w = message_loss(m_hat, m) if epoch >= params.reconstruct_pretrain_epochs else 0
        loss_i = image_loss(x_w, x0)  # b c h w -> 1
        loss_p = perceptual_loss(x_w, x0) if perceptual_loss is not None else 0
        loss = params.lambda_w * loss_w + params.lambda_i * loss_i + params.lambda_p * loss_p

        # stats
        ori_msgs = torch.sign(m) > 0
        decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
        bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
        word_accs = bit_accs == 1  # b
        log_stats = {
            'loss': itemize(loss),
            'loss_w': itemize(loss_w),
            'loss_i': itemize(loss_i),
            'loss_p': itemize(loss_p),
            'bit_acc': torch.mean(bit_accs).item(),
            'word_acc': torch.mean(word_accs.type(torch.float)).item(),
            **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()},
        }

        for name, attack in eval_attacks.items():
            m_hat, (x_w, x_r) = encoder_decoder(x0, m_normalized, eval_attack=attack)
            decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
            bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()

        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if (params.eval_only or epoch % params.saveimg_freq == 0) and it == 0 and utils.is_main_process():
            save_image(params.denormalize(x0),
                       os.path.join(params.imgs_dir, f'{epoch:03d}_val_x0.png'), nrow=8)
            save_image(params.denormalize(x_w),
                       os.path.join(params.imgs_dir, f'{epoch:03d}_val_xw.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

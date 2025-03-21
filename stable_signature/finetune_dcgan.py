import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden.ops import attacks as hidden_attacks, metrics as hidden_metrics
from hidden.models.attack_layers import HiddenAttackLayer
from stable_signature import utils
from stable_signature.models import hidden_utils
from stable_signature.models import dcgan
from stable_signature.models.resnet import resnet18
from stable_signature.loss.kl_div_softmax import KLDivSoftmaxLoss
from stable_signature.loss.loss_provider import LossProvider
from stable_signature.loss.matching_loss import MatchingLoss


def parse_args(verbose: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(__file__))

    g = parser.add_argument_group('Data parameters')
    g.add_argument('exp', type=str, nargs='?', default=None, help='Experiment name')
    g.add_argument('--dataset', type=str, default=None)
    g.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data'))
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
    g.add_argument('--img_size', type=int, default=256, help='Resize images to this size')
    g.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')

    g.add_argument('--loss_i', type=str, default='watson-vgg',
                   help='Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.')
    g.add_argument('--loss_w', type=str, default='bce',
                   help='Type of loss for the watermark loss. Can be mse or bce')
    g.add_argument('--loss_d', type=str, default='none',
                   help='Type of loss for the distillation loss. Can be kl, mse')
    g.add_argument('--lambda_i', type=float, default=1.0,
                   help='Weight of the image loss in the total loss')
    g.add_argument('--lambda_w', type=float, default=1.0,
                   help='Weight of the watermark loss in the total loss')
    g.add_argument('--lambda_d', type=float, default=1.0,
                   help='Weight of the distillation loss in the total loss')
    g.add_argument('--loss_i_dir', type=str, default=os.path.join(project_root, 'ckpts/loss'),
                   help='Pretrained weights dir for image loss.')
    g.add_argument('--optimizer', type=str, default='AdamW,lr=5e-4',
                   help='Optimizer and learning rate for training')
    g.add_argument('--steps', type=int, default=100,
                   help='Number of steps to train the model for')
    g.add_argument('--warmup_steps', type=int, default=20,
                   help='Number of warmup steps for the optimizer')

    g = parser.add_argument_group('Eval parameters')
    g.add_argument('--eval_steps', type=int, default=100,
                   help='Number of steps to evaluate the model for')
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
    g.add_argument('--output_dir', type=str, default='outputs',
                   help='Output directory for logs and images (Default: output)')
    g.add_argument('--seed', type=int, default=0)

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)

    if params.attack_layer is not None:
        if params.attack_layer.lower() == 'none':
            params.attack_layer = None
    if params.loss_d is not None:
        if params.loss_d.lower() == 'none':
            params.loss_d = None

    # Print the arguments
    if verbose:
        print('__git__:{}'.format(utils.get_sha()))
        print('__log__:{}'.format(json.dumps(vars(params))))

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

    # Create output dirs
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    params.imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(params.imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building Generator...')
    G0 = dcgan.get_generator(params.dataset, params.img_channels, params.z_dim).to(params.device)
    G0.load_state_dict(torch.load(params.generator_ckpt, weights_only=False, map_location=params.device))
    G0.eval()

    # Loads attack layer
    if params.attack_layer is None:
        attack_layer = None
    elif params.attack_layer == 'hidden':
        attack_layer = HiddenAttackLayer(
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
    print(f'Losses: {params.loss_w}, {params.loss_i}, and {params.loss_d}...')
    if params.loss_w == 'mse':
        message_loss = lambda m_hat, m, temp=10.0: torch.mean((m_hat * temp - (2 * m - 1)) ** 2)  # b k - b k
    elif params.loss_w == 'bce':
        message_loss = lambda m_hat, m, temp=10.0: torch.nn.functional.binary_cross_entropy_with_logits(
            m_hat * temp, m, reduction='mean')
    else:
        raise ValueError(f'Unknown message loss: {params.loss_w}')

    provider = LossProvider(params.loss_i_dir)
    colorspace = 'LA' if params.img_channels == 1 else 'RGB'
    if params.loss_i == 'mse':
        image_loss = lambda x_w, x0: torch.mean((x_w - x0) ** 2)
    elif params.loss_i == 'watson-dft':
        loss_perceptual = provider.get_loss_function(
            'Watson-DFT', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
        image_loss = lambda x_w, x0: loss_perceptual((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
    elif params.loss_i == 'watson-vgg':
        loss_perceptual = provider.get_loss_function(
            'Watson-VGG', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
        image_loss = lambda x_w, x0: loss_perceptual((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
    elif params.loss_i == 'ssim':
        loss_perceptual = provider.get_loss_function(
            'SSIM', colorspace=colorspace, pretrained=True, reduction='sum').to(params.device)
        image_loss = lambda x_w, x0: loss_perceptual((1 + x_w) / 2.0, (1 + x0) / 2.0) / x_w.size(0)
    else:
        raise ValueError(f'Unknown image loss: {params.loss_i}')

    if params.loss_d is None:
        distillation_loss = None
    else:
        # Load resnet18_clf model for distillation loss
        F_clf = resnet18(img_channels=params.img_channels, low_resolution=True).to(params.device)
        F_clf.load_state_dict(torch.load(params.clf_ckpt, weights_only=False, map_location=params.device))
        F_clf.eval()
        if params.loss_d == 'kl':
            distillation_loss = MatchingLoss(KLDivSoftmaxLoss(reduction='batchmean', dim=1), F_clf)
        elif params.loss_d == 'mse':
            distillation_loss = MatchingLoss(nn.MSELoss(reduction='mean'), F_clf)
        else:
            raise ValueError(f'Unknown distillation loss: {params.loss_d}')

    # attacks
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_03': hidden_attacks.CenterCrop(0.3),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'rot_25': hidden_attacks.Rotate(25),
        'rot_90': hidden_attacks.Rotate(90),
        'resize_03': hidden_attacks.Resize(0.3),
        'resize_07': hidden_attacks.Resize(0.7),
        'brightness_1p5': hidden_attacks.AdjustBrightness(
            1.5, mean=params.data_mean, std=params.data_std).to(params.device),
        'brightness_2': hidden_attacks.AdjustBrightness(
            2, mean=params.data_mean, std=params.data_std).to(params.device),
        'blur': hidden_attacks.GaussianBlur(kernel_size=5, sigma=0.5,
                                            mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_90': hidden_attacks.JPEGCompress(
            90, mean=params.data_mean, std=params.data_std).to(params.device),
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

        # Copy the Generator and finetune the copy
        G = deepcopy(G0).to(params.device)
        for param in G.parameters():
            param.requires_grad = True
        optim_params = utils.parse_initializer_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=G.parameters(), **optim_params)

        # Training loop
        print(f'>>> Training...')
        train_stats = train(optimizer, message_loss, image_loss, distillation_loss,
                            G0, G, attack_layer, msg_decoder, img_transform, key, metrics, params)
        val_stats = val(G0, G, msg_decoder, img_transform, key, eval_attacks, metrics, params)
        log_stats = {'key': key_str,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     }
        save_dict = {
            'generator': G.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        torch.save(save_dict, os.path.join(params.output_dir, f'checkpoint_{ii_key:03d}.pth'))
        with (Path(params.output_dir) / 'log.txt').open('a') as f:
            f.write(json.dumps(log_stats) + '\n')
        with (Path(params.output_dir) / 'keys.txt').open('a') as f:
            f.write(os.path.join(params.output_dir, f'checkpoint_{ii_key:03d}.pth') + '\t' + key_str + '\n')
        print('\n')


def itemize(tensor):
    if torch.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return tensor.item()
    return tensor


def train(optimizer: torch.optim.Optimizer, message_loss: Callable, image_loss: Callable, distillation_loss: Callable,
          G0: nn.Module, G: nn.Module, attack_layer: nn.Module, msg_decoder: nn.Module, img_transform,
          key: torch.Tensor, metrics: Dict, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger()
    G.train()

    base_lr = optimizer.param_groups[0]['lr']
    m = key.repeat(params.batch_size, 1)
    ori_msgs = torch.sign(m) > 0
    for it in metric_logger.log_every(range(1, params.steps + 1), params.log_freq, header):
        utils.adjust_learning_rate(optimizer, it, params.steps, params.warmup_steps, base_lr)
        # random latent vector
        z = torch.randn(params.batch_size, params.z_dim, 1, 1, device=params.device)  # b z 1 1
        # decode latents with original and fine-tuned decoder
        x0 = G0(z)  # b z 1 1 -> b c h w
        x_w = G(z)  # b z 1 1 -> b c h w

        # simulated attacks
        x_r = attack_layer(x_w, x0) if attack_layer is not None else x_w
        # extract watermark
        m_hat = msg_decoder(img_transform(x_r))  # b c h w -> b k

        # compute loss
        loss_w = message_loss(m_hat, m)
        loss_i = image_loss(x_w, x0)
        loss_d = distillation_loss(x_w, x0) if distillation_loss is not None else 0
        loss = params.lambda_w * loss_w + params.lambda_i * loss_i + params.lambda_d * loss_d

        # optim step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log stats
        decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
        bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
        word_accs = bit_accs == 1  # b
        log_stats = {
            'lr': optimizer.param_groups[0]['lr'],
            'loss': itemize(loss),
            'loss_w': itemize(loss_w),
            'loss_i': itemize(loss_i),
            'loss_d': itemize(loss_d),
            'bit_acc': torch.mean(bit_accs).item(),
            'word_acc': torch.mean(word_accs.float()).item(),
        }
        if params.log_train_metrics and it % 100 == 0:
            log_stats.update({
                **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()}
            })

        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})
        if it % params.log_freq == 0:
            print(json.dumps(log_stats))

        # save images during training
        if it % params.save_img_freq == 0:
            save_image(torch.clamp(params.denormalize(x0), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_train_x0.png'), nrow=8)
            save_image(torch.clamp(params.denormalize(x_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_train_xw.png'), nrow=8)

    print(f'✔️ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val(G0: nn.Module, G: nn.Module, msg_decoder: nn.Module, img_transform,
        key: torch.Tensor, eval_attacks: Dict, metrics: Dict, params: argparse.Namespace):
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
        x0 = G0(z)  # b z 1 1 -> b c h w
        x_w = G(z)  # b z 1 1 -> b c h w

        log_stats = {
            **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()},
        }
        for name, attack in eval_attacks.items():
            x_r = attack(img_transform(x_w))
            m_hat = msg_decoder(x_r)  # b c h w -> b k
            decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
            bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
            word_accs = (bit_accs == 1)  # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if it == 1:
            save_image(torch.clamp(params.denormalize(x0), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_val_x0.png'), nrow=8)
            save_image(torch.clamp(params.denormalize(x_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_val_xw.png'), nrow=8)

    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

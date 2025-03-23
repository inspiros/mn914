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
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden.ops import attacks as hidden_attacks, metrics as hidden_metrics
from hidden.models import attack_layers
from stable_signature import utils
from stable_signature.models import hidden_utils
from stable_signature.models import dcgan


def parse_args(verbose: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Data parameters')
    g.add_argument('exp', type=str, nargs='?', default=None, help='Experiment name')
    g.add_argument('--data_mean', type=utils.tuple_inst(float), default=None)
    g.add_argument('--data_std', type=utils.tuple_inst(float), default=None)

    g = parser.add_argument_group('Model parameters')
    g.add_argument('--generator_ckpt', type=str, required=True,
                   help='Path to the checkpoint file for the Generator')
    g.add_argument('--decoder', type=str, choices=['hidden', 'resnet'], default='hidden',
                   help='Decoder type')
    g.add_argument('--decoder_path', type=str, required=True,
                   help='Path to the hidden decoder for the watermarking model')
    g.add_argument('--num_bits', type=int, default=16, help='Number of bits in the watermark')
    g.add_argument('--z_dim', type=int, default=100, help='Dimension of the latent vector')
    g.add_argument('--decoder_depth', type=int, default=8,
                   help='Depth of the decoder in the watermarking model')
    g.add_argument('--decoder_channels', type=int, default=64,
                   help='Number of channels in the decoder of the watermarking model')

    g = parser.add_argument_group('Eval parameters')
    g.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    g.add_argument('--img_size', type=int, default=256, help='Resize images to this size')
    g.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    g.add_argument('--eval_steps', type=int, default=100,
                   help='Number of steps to evaluate the model for')

    g = parser.add_argument_group('Distributed training parameters')
    g.add_argument('--device', type=str, default='cuda:0', help='Device')

    g = parser.add_argument_group('Experiments parameters')
    g.add_argument('--output_dir', type=str, default='outputs',
                   help='Output directory for logs and images (Default: output)')
    g.add_argument('--seed', type=int, default=0)

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)

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

    # Loads LDM auto-encoder models
    print(f'>>> Building Generator...')
    G0 = dcgan.Generator(params.img_channels, params.z_dim).to(params.device)
    G0.load_state_dict(torch.load(params.generator_ckpt, weights_only=False, map_location=params.device))
    G0.eval()

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

    # attacks
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_08': hidden_attacks.CenterCrop(0.8),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'resize2_08': hidden_attacks.Resize2(0.8),
        'resize2_05': hidden_attacks.Resize2(0.5),
        'rot_r25': hidden_attacks.Rotate(25, fill=-1),
        'rot_l25': hidden_attacks.Rotate(-25, fill=-1),
        'rot_r': hidden_attacks.Rotate(90),
        'rot_l': hidden_attacks.Rotate(-90),
        'hflip': hidden_attacks.HFlip(),
        'vflip': hidden_attacks.VFlip(),
        'brightness_d': hidden_attacks.AdjustBrightness(
            0.75, mean=params.data_mean, std=params.data_std).to(params.device),
        'brightness_i': hidden_attacks.AdjustBrightness(
            1.25, mean=params.data_mean, std=params.data_std).to(params.device),
        'contrast_d': hidden_attacks.AdjustContrast(
            0.75, mean=params.data_mean, std=params.data_std).to(params.device),
        'contrast_i': hidden_attacks.AdjustContrast(
            1.25, mean=params.data_mean, std=params.data_std).to(params.device),
        'saturation_d': hidden_attacks.AdjustSaturation(
            0.75, mean=params.data_mean, std=params.data_std).to(params.device),
        'saturation_i': hidden_attacks.AdjustSaturation(
            1.25, mean=params.data_mean, std=params.data_std).to(params.device),
        'hue_d': hidden_attacks.AdjustHue(
            -0.1, mean=params.data_mean, std=params.data_std).to(params.device),
        'hue_i': hidden_attacks.AdjustHue(
            0.1, mean=params.data_mean, std=params.data_std).to(params.device),
        'sharpness_d': hidden_attacks.AdjustSharpness(
            0.75, mean=params.data_mean, std=params.data_std).to(params.device),
        'sharpness_i': hidden_attacks.AdjustSharpness(
            1.25, mean=params.data_mean, std=params.data_std).to(params.device),
        'blur': hidden_attacks.GaussianBlur(
            kernel_size=5, sigma=0.5, mean=params.data_mean, std=params.data_std).to(params.device),
        'posterize_7': hidden_attacks.Posterize(7),
        'posterize_6': hidden_attacks.Posterize(6),
        'posterize_5': hidden_attacks.Posterize(5),
        'autocontrast': hidden_attacks.AutoContrast(mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_80': hidden_attacks.JPEGCompress(
            80, mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg_50': hidden_attacks.JPEGCompress(
            50, mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg2000_80': hidden_attacks.JPEG2000Compress(
            80, mean=params.data_mean, std=params.data_std).to(params.device),
        'jpeg2000_50': hidden_attacks.JPEG2000Compress(
            50, mean=params.data_mean, std=params.data_std).to(params.device),
        'webp_80': hidden_attacks.WEBPCompress(
            80, mean=params.data_mean, std=params.data_std).to(params.device),
        'webp_50': hidden_attacks.WEBPCompress(
            50, mean=params.data_mean, std=params.data_std).to(params.device),
    }
    if params.img_channels == 1:
        # remove saturation and hue
        eval_attacks = {k: v for k, v in eval_attacks.items()
                        if not (k.startswith('saturation') or k.startswith('hue'))}
        # add morphology
        morphology_footprint = hidden_attacks.morphology.footprints.diamond(1)
        eval_attacks.update({
            'erosion': hidden_attacks.morphology.Erosion(
                morphology_footprint, mean=params.data_mean, std=params.data_std).to(params.device),
            'dilation': hidden_attacks.morphology.Dilation(
                morphology_footprint, mean=params.data_mean, std=params.data_std).to(params.device),
            'opening': hidden_attacks.morphology.Opening(
                morphology_footprint, mean=params.data_mean, std=params.data_std).to(params.device),
            'closing': hidden_attacks.morphology.Closing(
                morphology_footprint, mean=params.data_mean, std=params.data_std).to(params.device),
        })
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

    # Create output dirs
    params.output_dir = os.path.join(params.output_dir, 'eval')
    os.makedirs(params.output_dir, exist_ok=True)
    params.imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(params.imgs_dir, exist_ok=True)

    # Eval
    models = []
    with (Path(params.output_dir) / '../keys.txt').open('r') as f:
        for line in map(str.strip, f.readlines()):
            if len(line):
                models.append(line.split())
    for ii_key in range(len(models)):
        model_id = models[ii_key][0]
        key_str = models[ii_key][1]
        model_path = os.path.join(params.output_dir, '..', f'checkpoint_{model_id}.pth')
        key = str2msg(key_str, device=params.device)
        print(f'Key: {key_str}, model: {model_path}')

        # Copy the Generator and load weights
        G = deepcopy(G0).to(params.device)
        G.eval()
        G.load_state_dict(torch.load(model_path, weights_only=False, map_location=params.device)['generator'])

        print(f'>>> Evaluating...')
        val_stats = val(G0, G, msg_decoder, img_transform, key, eval_attacks, metrics, params)
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}
        # save
        with (Path(params.output_dir) / f'log_{ii_key:03d}.csv').open('a') as f:
            for i, k in enumerate(log_stats.keys()):
                f.write(k)
                if i != len(log_stats) - 1:
                    f.write(',')
            f.write('\n')
            for i, v in enumerate(log_stats.values()):
                f.write(str(v))
                if i != len(log_stats) - 1:
                    f.write(',')
        with (Path(params.output_dir) / f'log_{ii_key:03d}.txt').open('a') as f:
            f.write(json.dumps(log_stats) + '\n')
        print('\n')


def str2msg(m_str: str, device=None) -> torch.Tensor:
    return torch.tensor([True if el == '1' else False for el in m_str], device=device)


@torch.no_grad()
def val(G0: nn.Module, G: nn.Module, msg_decoder: nn.Module, img_transform,
        key: torch.Tensor, eval_attacks: Dict, metrics: Dict, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger()
    G.eval()

    m = key.repeat(params.batch_size, 1)
    ori_msgs = torch.sign(m) > 0
    # assuring same latent vectors generated
    generator = torch.Generator(device=params.device).manual_seed(params.seed)
    for it in metric_logger.log_every(range(1, params.eval_steps + 1), 10, header):
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
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if it <= 10:
            save_image(torch.clamp(params.denormalize(x0), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_val_x0.png'), nrow=8)
            save_image(torch.clamp(params.denormalize(x_w), 0, 1),
                       os.path.join(params.imgs_dir, f'{it:05d}_val_xw.png'), nrow=8)

    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

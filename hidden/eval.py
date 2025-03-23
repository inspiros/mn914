import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image

# add hidden path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidden import models, utils
from hidden.models import attenuations, attack_layers
from hidden.ops import attacks as hidden_attacks, metrics as hidden_metrics, transforms as hidden_transforms


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
    g.add_argument('--scaling_w', type=float, default=0.3,
                   help='Scaling of the watermark signal. (Default: 0.3)')
    g.add_argument('--scaling_i', type=float, default=1.0,
                   help='Scaling of the original image. (Default: 1.0)')

    g = parser.add_argument_group('Loader parameters')
    g.add_argument('--batch_size', type=int, default=64, help='Batch size. (Default: 64)')
    g.add_argument('--workers', type=int, default=8,
                   help='Number of workers for data loading. (Default: 8)')

    g = parser.add_argument_group('Attenuation parameters')
    g.add_argument('--attenuation', type=utils.nullable(str), default=None,
                   help='Attenuation type. (Default: None)')
    g.add_argument('--scale_channels', type=utils.bool_inst, default=False,
                   help='Use channel scaling. (Default: False)')

    g = parser.add_argument_group('Distributed training parameters')
    g.add_argument('--device', type=str, default='cuda:0', help='Device')

    g = parser.add_argument_group('Misc')
    g.add_argument('--seed', default=0, type=int, help='Random seed')

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)
    if (params.data_mean is None) ^ (params.data_std is None):
        raise ValueError('Data mean and std are both required.')

    # Print the arguments
    if verbose:
        print(params)

    return params


def main():
    params = parse_args()

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
        val_loader = utils.get_dataloader(params.data_dir, dataset=params.dataset, train=False,
                                          transform=val_transform, batch_size=params.batch_size,
                                          num_workers=params.workers, shuffle=False)
    else:
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
        decoder = models.resnet18_decoder(num_bits=params.num_bits,
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

    # attacks
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_08': hidden_attacks.CenterCrop(0.8),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'resize_08': hidden_attacks.Resize(0.8),
        'resize_05': hidden_attacks.Resize(0.5),
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
        for k in eval_attacks.keys():
            if k.startswith('saturation') or k.startswith('hue'):
                eval_attacks.pop(k)
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

    # Create encoder/decoder
    encoder_decoder = models.EncoderDecoder(encoder=encoder,
                                            attenuation=attenuation,
                                            attack_layer=None,
                                            decoder=decoder,
                                            generate_delta=params.generate_delta,
                                            scale_channels=params.scale_channels,
                                            scaling_i=params.scaling_i,
                                            scaling_w=params.scaling_w,
                                            num_bits=params.num_bits,
                                            std=params.data_std if params.scale_channels else None)
    encoder_decoder = encoder_decoder.to(params.device)

    # optionally resume training
    to_restore = {'epoch': 1}
    if os.path.isfile(os.path.join(params.output_dir, 'checkpoint.pth')):
        utils.restart_from_checkpoint(
            os.path.join(params.output_dir, 'checkpoint.pth'),
            run_variables=to_restore,
            encoder_decoder=encoder_decoder,
        )
    elif params.resume_from is not None:
        utils.restart_from_checkpoint(
            params.resume_from,
            encoder_decoder=encoder_decoder
        )
        if params.encoder_only_epochs:
            decoder.requires_grad_(False)
    start_epoch = to_restore['epoch']

    # create output dir
    params.output_dir = os.path.join(params.output_dir, 'eval')
    os.makedirs(params.output_dir, exist_ok=True)
    print('evaluating...')
    val_stats = eval_one_epoch(encoder_decoder, val_loader, eval_attacks, metrics, params)
    log_stats = {'epoch': start_epoch, **{f'val_{k}': v for k, v in val_stats.items()}}
    # save
    with (Path(params.output_dir) / 'log.csv').open('a') as f:
        for i, k in enumerate(log_stats.keys()):
            f.write(k)
            if i != len(log_stats) - 1:
                f.write(',')
        f.write('\n')
        for i, v in enumerate(log_stats.values()):
            f.write(str(v))
            if i != len(log_stats) - 1:
                f.write(',')
    with (Path(params.output_dir) / 'log.txt').open('a') as f:
        f.write(json.dumps(log_stats) + '\n')


# noinspection DuplicatedCode
@torch.no_grad()
def eval_one_epoch(encoder_decoder: models.EncoderDecoder, loader, eval_attacks, metrics, params):
    r"""
    One epoch of eval.
    """
    header = '[Eval]'
    encoder_decoder.eval()
    metric_logger = utils.MetricLogger()

    # assuring same keys generated
    generator = torch.Generator(device=params.device).manual_seed(params.seed)
    for it, (x0, _) in enumerate(metric_logger.log_every(loader, 10, f'- {header}')):
        x0 = x0.to(params.device, non_blocking=True)  # b c h w

        m = torch.randint(0, 2, (x0.size(0), params.num_bits), dtype=torch.float32, device=params.device,
                          generator=generator)  # b k [0 1]
        m_normalized = 2 * m - 1  # b k [-1 1]

        m_hat, (x_w, x_r) = encoder_decoder(x0, m_normalized)

        # stats
        ori_msgs = torch.sign(m) > 0
        decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
        bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
        word_accs = bit_accs == 1  # b
        log_stats = {
            'bit_acc': torch.mean(bit_accs).item(),
            'word_acc': torch.mean(word_accs.float()).item(),
            **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()},
        }

        for name, attack in eval_attacks.items():
            m_hat, (x_w, x_r) = encoder_decoder(x0, m_normalized, eval_attack=attack)
            decoded_msgs = torch.sign(m_hat) > 0  # b k -> b k
            bit_accs = torch.sum(ori_msgs == decoded_msgs, dim=-1) / m.size(1)  # b k -> b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()

        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if it == 0:
            save_image(params.denormalize(x0),
                       os.path.join(params.imgs_dir, f'{it:03d}_val_x0.png'), nrow=8)
            save_image(params.denormalize(x_w),
                       os.path.join(params.imgs_dir, f'{it:03d}_val_xw.png'), nrow=8)

    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

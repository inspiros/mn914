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
from hidden.models import attenuations
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
    g.add_argument('--output_dir', type=str, default='outputs_eval',
                   help='Output directory for logs and images (Default: outputs_eval)')
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
    g.add_argument('--resume_from', default=None, type=str,
                   help='Checkpoint path to resume from.')
    g.add_argument('--scaling_w', type=float, default=1.0,
                   help='Scaling of the watermark signal. (Default: 1.0)')
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
    g.add_argument('--seed', default=0, type=utils.nullable(int), help='Random seed')

    params = parser.parse_args()

    if params.exp is not None:
        params.output_dir = os.path.join(params.output_dir, params.exp)
    if (params.data_mean is None) ^ (params.data_std is None):
        raise ValueError('Data mean and std are both required.')

    # Print the arguments
    if verbose:
        print('git:', utils.get_sha())
        print(json.dumps(vars(params)))

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
    utils.restart_from_checkpoint(
        params.resume_from,
        encoder_decoder=encoder_decoder
    )
    # create output dir
    os.makedirs(params.output_dir, exist_ok=True)

    # eval
    print('evaluating...')
    val_stats = eval_one_epoch(encoder_decoder.encoder_wrapper(), train_loader, metrics, params)
    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}
    with (Path(params.output_dir) / 'log.txt').open('a') as f:
        f.write(json.dumps(log_stats) + '\n')


def itemize(tensor):
    if torch.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return tensor.item()
    return tensor


# noinspection DuplicatedCode
@torch.no_grad()
def eval_one_epoch(encoder: models.EncoderWithJND, loader, metrics, params):
    r"""
    One epoch of eval.
    """
    header = f'[Eval]'
    encoder.eval()
    metric_logger = utils.MetricLogger()

    # assuring same keys generated
    generator = torch.Generator(device=params.device).manual_seed(params.eval_seed)
    for it, (x0, _) in enumerate(metric_logger.log_every(loader, 10, f'- {header}')):
        x0 = x0.to(params.device, non_blocking=True)  # b c h w

        m = torch.randint(0, 2, (x0.size(0), params.num_bits), dtype=torch.float32, device=params.device,
                          generator=generator)  # b k [0 1]
        m_normalized = 2 * m - 1  # b k [-1 1]

        x_w = encoder(x0, m_normalized)

        # stats
        log_stats = {
            **{metric_name: metric(x_w, x0).mean().item() for metric_name, metric in metrics.items()},
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        if it < 10:
            save_image(params.denormalize(x0),
                       os.path.join(params.imgs_dir, f'{it:03d}_val_x0.png'), nrow=8)
            save_image(params.denormalize(x_w),
                       os.path.join(params.imgs_dir, f'{it:03d}_val_xw.png'), nrow=8)

    print(f'⭕ {header}', metric_logger, end='\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()

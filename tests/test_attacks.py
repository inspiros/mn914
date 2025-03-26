import os
import shutil

import kornia.augmentation as K
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from hidden.models import attack_layers
from hidden.ops import attacks as hidden_attacks


def test_attack_layer(n_samples: int = 5):
    img = Image.open('../hidden/images/00.png').resize((32, 32)).convert('RGB')
    x = transforms.ToTensor()(img)
    x = x * 2 - 1
    x = x.unsqueeze(0)
    x = x.repeat(n_samples, 1, 1, 1).clone()
    x.requires_grad_(True)
    print('input.shape:', x.shape)

    attack_layer = attack_layers.HiddenAttackLayer(img.size)

    out_dir = 'outputs/attack_layer'
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(attack_layer.augmentations)):
        x_r = attack_layer.forward_debug(x, torch.ones_like(x), i)
        print('output.shape:', x_r.shape)
        x_r.sum().backward()
        assert x.grad is not None and x.grad.ne(0).any()
        x.grad = None
        save_image((x_r * 0.5 + 0.5).detach().cpu(), os.path.join(out_dir, f'{i:02d}.png'))


def test_rgb_attacks():
    img = Image.open('../hidden/images/00.png').resize((32, 32)).convert('RGB')
    x = transforms.ToTensor()(img)
    x = x * 2 - 1
    x = x.unsqueeze(0)
    print('input.shape:', x.shape)

    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_08': hidden_attacks.CenterCrop(0.8),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'rot_r25': hidden_attacks.Rotate(25, fill=-1),
        'rot_l25': hidden_attacks.Rotate(-25, fill=-1),
        'rot_l': hidden_attacks.Rotate(90),
        'rot_r': hidden_attacks.Rotate(270),
        'hflip': hidden_attacks.HFlip(),
        'vflip': hidden_attacks.VFlip(),
        'resize_08': hidden_attacks.Resize(0.8),
        'resize_05': hidden_attacks.Resize(0.5),
        'resize2_08': hidden_attacks.Resize2(0.8),
        'resize2_05': hidden_attacks.Resize2(0.5),
        'brightness_d': hidden_attacks.AdjustBrightness(0.75),
        'brightness_i': hidden_attacks.AdjustBrightness(1.25),
        'contrast_d': hidden_attacks.AdjustContrast(0.75),
        'contrast_i': hidden_attacks.AdjustContrast(1.25),
        'saturation_d': hidden_attacks.AdjustSaturation(0.75),
        'saturation_i': hidden_attacks.AdjustSaturation(1.25),
        'hue_d': hidden_attacks.AdjustHue(-0.1),
        'hue_i': hidden_attacks.AdjustHue(0.1),
        'sharpness_d': hidden_attacks.AdjustSharpness(0.75),
        'sharpness_i': hidden_attacks.AdjustSharpness(1.25),
        'blur': hidden_attacks.GaussianBlur(kernel_size=5, sigma=0.5),
        'invert': hidden_attacks.Invert(),
        'posterize_7': hidden_attacks.Posterize(7),
        'posterize_6': hidden_attacks.Posterize(6),
        'posterize_5': hidden_attacks.Posterize(5),
        'solarize': hidden_attacks.Solarize(0.9),
        'autocontrast': hidden_attacks.AutoContrast(),
        'jpeg_80': hidden_attacks.JPEGCompress(80),
        'jpeg_50': hidden_attacks.JPEGCompress(50),
        'jpeg2000_80': hidden_attacks.JPEG2000Compress(80),
        'jpeg2000_50': hidden_attacks.JPEG2000Compress(50),
        'webp_80': hidden_attacks.WEBPCompress(80),
        'webp_50': hidden_attacks.WEBPCompress(50),
        'watermark_dropout': hidden_attacks.WatermarkDropout(0.2),
        'watermark_cropout': hidden_attacks.WatermarkCenterCropout(0.75),
    }
    out_dir = 'outputs/rgb_attacks'
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, (name, attack) in enumerate(eval_attacks.items()):
        x_r = attack(x) if not name.startswith('watermark') else attack(x, torch.zeros_like(x))
        print(name, x_r.shape)
        save_image((x_r * 0.5 + 0.5).detach().cpu(), os.path.join(out_dir, f'{i}_{name}.png'))


def test_grayscale_attacks():
    img = Image.open('../hidden/images/3.png').resize((28, 28)).convert('L')
    x = transforms.ToTensor()(img)
    x = x * 2 - 1
    x = x.unsqueeze(0)
    print('input.shape:', x.shape)

    morphology_footprint = hidden_attacks.morphology.footprints.diamond(1)
    eval_attacks = {
        'none': hidden_attacks.Identity(),
        'crop_08': hidden_attacks.CenterCrop(0.8),
        'crop_05': hidden_attacks.CenterCrop(0.5),
        'rot_r25': hidden_attacks.Rotate(25, fill=-1),
        'rot_l25': hidden_attacks.Rotate(-25, fill=-1),
        'rot_l': hidden_attacks.Rotate(90),
        'rot_r': hidden_attacks.Rotate(270),
        'hflip': hidden_attacks.HFlip(),
        'vflip': hidden_attacks.VFlip(),
        'resize_08': hidden_attacks.Resize(0.8),
        'resize_05': hidden_attacks.Resize(0.5),
        'resize2_08': hidden_attacks.Resize2(0.8),
        'resize2_05': hidden_attacks.Resize2(0.5),
        'brightness_d': hidden_attacks.AdjustBrightness(0.75),
        'brightness_i': hidden_attacks.AdjustBrightness(1.25),
        'contrast_d': hidden_attacks.AdjustContrast(0.75),
        'contrast_i': hidden_attacks.AdjustContrast(1.25),
        'sharpness_d': hidden_attacks.AdjustSharpness(0.75),
        'sharpness_i': hidden_attacks.AdjustSharpness(1.25),
        'blur': hidden_attacks.GaussianBlur(kernel_size=5, sigma=0.5),
        'invert': hidden_attacks.Invert(),
        'posterize_7': hidden_attacks.Posterize(7),
        'posterize_6': hidden_attacks.Posterize(6),
        'solarize': hidden_attacks.Solarize(0.9),
        'autocontrast': hidden_attacks.AutoContrast(),
        'jpeg_80': hidden_attacks.JPEGCompress(80),
        'jpeg_50': hidden_attacks.JPEGCompress(50),
        'jpeg2000_80': hidden_attacks.JPEG2000Compress(80),
        'jpeg2000_50': hidden_attacks.JPEG2000Compress(50),
        'webp_80': hidden_attacks.WEBPCompress(80),
        'webp_50': hidden_attacks.WEBPCompress(50),
        'watermark_dropout': hidden_attacks.WatermarkDropout(0.2),
        'watermark_cropout': hidden_attacks.WatermarkCenterCropout(0.75),
        # morphology
        'erosion': hidden_attacks.morphology.Erosion(morphology_footprint),
        'dilation': hidden_attacks.morphology.Dilation(morphology_footprint),
        'opening': hidden_attacks.morphology.Opening(morphology_footprint),
        'closing': hidden_attacks.morphology.Closing(morphology_footprint),
        'white_tophat': hidden_attacks.morphology.WhiteTophat(morphology_footprint),
        'black_tophat': hidden_attacks.morphology.BlackTophat(morphology_footprint),
    }
    out_dir = 'outputs/grayscale_attacks'
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, (name, attack) in enumerate(eval_attacks.items()):
        x_r = attack(x) if not name.startswith('watermark') else attack(x, torch.zeros_like(x))
        print(name, x_r.shape)
        save_image((x_r * 0.5 + 0.5).detach().cpu(), os.path.join(out_dir, f'{i}_{name}.png'))


def test_kornia(n_samples: int = 5):
    img = Image.open('../hidden/images/00.png').resize((256, 256)).convert('RGB')
    x = transforms.ToTensor()(img)
    # x = x * 2 - 1
    x = x.unsqueeze(0)
    x = x.repeat(n_samples, 1, 1, 1)
    print('input.shape:', x.shape)

    # aug = K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.5, 4.))
    # aug = random_attacks.RandomSizedCrop(size=(0.2, 1.), p=1)
    # aug = random_attacks.RandomCenterCrop(output_size=((0.5, 1.), (0.5, 1.)), p=1)
    # aug = K.RandomAffine(degrees=(-45, 45), p=1)
    # aug = K.RandomCrop(size=(15, 15), p=1)
    aug = K.RandomResizedCrop(size=img.size, scale=(0.2, 1.0), p=1)
    x_aug = aug(x)
    print('output.shape:', x_aug.shape)
    save_image(x_aug, 'test.png')


if __name__ == '__main__':
    test_attack_layer()
    test_rgb_attacks()
    test_grayscale_attacks()
    # test_kornia()

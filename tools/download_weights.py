"""
Script to download pre-trained weights.
"""
import argparse
import os
import urllib.request

import gdown


def parse_args():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default=os.path.join(project_root, 'ckpts'))
    parser.add_argument('--download_r3gan', action='store_true', default=False)
    return parser.parse_args()


def download_from_url(url, path, skip_existing=False):
    if not skip_existing or not os.path.isfile(path):
        print(f'Downloading {url}')
        urllib.request.urlretrieve(url, path)


def main():
    args = parse_args()
    ckpts_path = args.ckpt_dir
    os.makedirs(ckpts_path, exist_ok=True)
    loss_ckpts_path = os.path.join(ckpts_path, 'loss')
    os.makedirs(loss_ckpts_path, exist_ok=True)

    # HiDDeN pretrained on COCO
    ss_hidden_repo = 'https://github.com/facebookresearch/stable_signature/raw/refs/heads/main/hidden/ckpts'
    ss_hidden_weight = 'hidden_replicate.pth'
    download_from_url(ss_hidden_repo + '/' + ss_hidden_weight,
                      os.path.join(ckpts_path, ss_hidden_weight),
                      skip_existing=True)

    # Watson Perceptual Loss
    watson_repo_path = 'https://github.com/SteffenCzolbe/PerceptualSimilarity/raw/refs/heads/master/src/loss/weights'
    watson_weights = [
        'gray_adaptive_trial0.pth',
        'gray_pnet_lin_squeeze_trial0.pth',
        'gray_pnet_lin_vgg_trial0.pth',
        'gray_watson_dct_trial0.pth',
        'gray_watson_fft_trial0.pth',
        'gray_watson_vgg_trial0.pth',
        'rgb_adaptive_trial0.pth',
        'rgb_pnet_lin_squeeze_trial0.pth',
        'rgb_pnet_lin_vgg_trial0.pth',
        'rgb_watson_dct_trial0.pth',
        'rgb_watson_fft_trial0.pth',
        'rgb_watson_vgg_trial0.pth',
    ]
    for weight_file in watson_weights:
        download_from_url(watson_repo_path + '/' + weight_file,
                          os.path.join(loss_ckpts_path, weight_file),
                          skip_existing=True)

    # r3gan
    if args.download_r3gan:
        download_from_url('https://huggingface.co/brownvc/R3GAN-CIFAR10/resolve/main/network-snapshot-final.pkl?download=true',
                          os.path.join(ckpts_path, 'r3gan_cifar10.pkl'),
                          skip_existing=True)
        # download_from_url('https://huggingface.co/brownvc/R3GAN-ImgNet-64x64/resolve/main/network-snapshot-final.pkl?download=true',
        #                   os.path.join(ckpts_path, 'r3gan_imgnet_64x64.pkl'),
        #                   skip_existing=True)

    # the rest
    ckpts_drive_url = 'https://drive.google.com/drive/folders/1NmnhY8MAXJfFrAItFcJq_13nX59OqxM0?usp=drive_link'
    gdown.download_folder(ckpts_drive_url, output=ckpts_path)


if __name__ == '__main__':
    main()

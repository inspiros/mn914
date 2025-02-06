Revisited Stable Signature
------

This code was modified from
[Stable Signature codebase](https://github.com/facebookresearch/stable_signature).

## Setup

### Perceptual Losses

Before you can train with the Watson's perceptual losses, you need to download the weights from
[this link](https://github.com/SteffenCzolbe/PerceptualSimilarity/tree/master/src/loss/weights)
and then put them on the folder `../ckpts/loss` (or define the path with `--loss_i_dir`).

## Usage

### Fine-tuning GAN:

Instead of fine-tuning the decoder of a LDM as in the original work, we can finetune the generator of a GAN.
In this case, the notions of dataset and epoch are discarded as we can simply sample from the prior distribution.

<p align="center">
<img src="../resources/gan_basic_pipeline.png" height="300"/>
</p>

To use the modified pipeline:
- Enabling **simulated attack layer**: run the script with `--attack_layer hidden`.
- Enabling **distillation loss**: _T.B.D._
- Enabling **critic loss** (using the discriminator): _T.B.D._

<p align="center">
<img src="../resources/gan_pipeline.png" height="300"/>
</p>

The full parameters list is to be decided.

#### MNIST Example

- Linux:
```cmd
python finetune_gan.py mnist --num_keys 1 \
    --dataset MNIST --data_mean [0.5] --data_std [0.5] \
    --img_size 28 --img_channels 1 --num_bits 16 --batch_size 128 \
    --steps 1000 --eval_steps 10 \
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth \
    --msg_decoder_path ../ckpts/hidden_mnist.pth \
    --clf_ckpt ../ckpts/clf/resnet18_mnist.pth \
    --loss_i watson-vgg --loss_w bce --loss_d kl
```
- Windows:
```cmd
python finetune_gan.py mnist --num_keys 1 `
    --dataset MNIST --data_mean [0.5] --data_std [0.5] `
    --img_size 28 --img_channels 1 --num_bits 16 --batch_size 128 `
    --steps 1000 --eval_steps 10 `
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth `
    --msg_decoder_path ../ckpts/hidden_mnist.pth `
    --clf_ckpt ../ckpts/clf/resnet18_mnist.pth `
    --loss_i watson-vgg --loss_w bce --loss_d kl
```

#### CIFAR10 Example

- Linux:
```cmd
python finetune_gan.py cifar10 --num_keys 1 \
    --dataset CIFAR10 --data_mean [0.485,0.456,0.406] --data_std [0.229,0.224,0.225] \
    --img_size 32 --img_channels 3 --num_bits 16 --batch_size 128 \
    --steps 1000 --eval_steps 10 \
    --generator_ckpt ../ckpts/dcgan_generator_cifar10.pth \
    --msg_decoder_path ../ckpts/hidden_cifar10.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --loss_i watson-vgg --loss_w bce --loss_d kl
```
- Windows:
```cmd
python finetune_gan.py cifar10 --num_keys 1 `
    --dataset CIFAR10 --data_mean [0.485,0.456,0.406] --data_std [0.229,0.224,0.225] `
    --img_size 32 --img_channels 3 --num_bits 16 --batch_size 128 `
    --steps 1000 --eval_steps 10 `
    --generator_ckpt ../ckpts/dcgan_generator_cifar10.pth `
    --msg_decoder_path ../ckpts/hidden_cifar10.pth `
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth `
    --loss_i watson-vgg --loss_w bce --loss_d kl
```

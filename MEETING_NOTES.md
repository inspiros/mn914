## Notes for meeting on 28/02

### 1. Setup

These commands will clone the repository, install libraries and download pre-trained weights:

```cmd
git clone https://github.com/inspiros/mn914
cd mn914
pip install -r requirements.txt
python tools/download_weights.py
```

### 2. Run

#### 2.1. HiDDeN using U-Net-based watermark encoder

First, change working directory to [`hidden`](hidden):

```cmd
cd hidden
```

Then run:

```cmd
python main.py cifar10_unet \
    --dataset CIFAR10 --img_size 32 --img_channels 3 \
    --num_bits 32 --batch_size 256 --epochs 200 --eval_freq 5 \
    --encoder unet --generate_delta False \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 \
    --optimizer Lamb,lr=2e-2 \
    --loss_w bce --loss_i mse --lambda_w 1 --lambda_i 1
```

#### 2.2. Finetune GAN on CIFAR10 with Different Configurations

First, change working directory to [`stable_signature`](stable_signature):

```cmd
cd stale_signature
```

From this week, we migrate to R3GAN instead of DCGAN.
The watermark extractor used is the pretrained weights published by the authors.

##### 2.2.1. Baseline

```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --dataset CIFAR10 --img_size 32 \
    --num_bits 48 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --msg_decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer none \
    --loss_i watson-vgg --loss_w bce --loss_d none \
    --lambda_i 1 --lambda_w 0.05 --lambda_d 0
```

##### 2.2.2. With attack layer

```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --dataset CIFAR10 --img_size 32 \
    --num_bits 48 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --msg_decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer hidden \
    --loss_i watson-vgg --loss_w bce --loss_d none \
    --lambda_i 1 --lambda_w 0.05 --lambda_d 0
```

##### 2.2.3. With logit loss

```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --dataset CIFAR10 --img_size 32 \
    --num_bits 48 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --msg_decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer none \
    --loss_i watson-vgg --loss_w bce --loss_d kl \
    --lambda_i 1 --lambda_w 0.05 --lambda_d 0.1
```

##### 2.2.4. With both modifications

```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --dataset CIFAR10 --img_size 32 \
    --num_bits 48 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --msg_decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer hidden \
    --loss_i watson-vgg --loss_w bce --loss_d kl \
    --lambda_i 1 --lambda_w 0.05 --lambda_d 0.1
```

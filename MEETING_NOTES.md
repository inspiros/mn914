## Meeting Notes

### Setup

These commands will clone the repository, install libraries and download pre-trained weights:

```cmd
git clone https://github.com/inspiros/mn914
cd mn914
pip install -r requirements.txt
python tools/download_weights.py
```

### Run

#### HiDDeN

###### HiDDeN on MNIST

```cmd
python main.py mnist \
    --dataset MNIST --img_size 28 --img_channels 1 \
    --num_bits 32 --batch_size 16 --epochs 200 --eval_freq 5 \
    --encoder hidden --generate_delta True --decoder hidden \
    --optimizer Lamb,lr=2e-2 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=200,warmup_lr_init=1e-6,warmup_t=5 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w bce --loss_i mse --loss_p watson-vgg \
    --lambda_w 1 --lambda_i 0.02 --lambda_p 0.8
```

###### U-Net on COCO

```cmd
python main.py cifar10_unet \
    --dataset CIFAR10 --img_size 32 --img_channels 3 \
    --num_bits 48 --batch_size 32 --epochs 200 --eval_freq 5 \
    --encoder unet --generate_delta False \
    --decoder resnet --resume_from ../ckpts/hidden_resnet.pth \
    --encoder_only_epochs 10 \
    --optimizer Lamb,lr=2e-2 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=200,warmup_lr_init=1e-6,warmup_t=5 \
    --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 \
    --loss_w bce --loss_i mse --loss_p watson-vgg \
    --lambda_w 1 --lambda_i 0.02 --lambda_p 0.5
```

#### Stable Signature

###### on MNIST
All experiments use DCGAN.

- Baseline
```cmd
python finetune_dcgan.py mnist_baseline --num_keys 1 `
    --num_bits 32 --img_size 28 --img_channels 1 --batch_size 32 `
    --steps 8000 --eval_steps 400 --eval_freq 800 `
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth `
    --decoder_path ../ckpts/hidden_mnist.pth `
    --attack_layer none `
    --loss_i watson-vgg --loss_w bce --loss_c none --loss_d none `
    --lambda_i 1 --lambda_w 1 --lambda_c 0 --lambda_d 0
```

- With attack layer
```cmd
python finetune_dcgan.py mnist_attack --num_keys 1 `
    --num_bits 32 --img_size 28 --img_channels 1 --batch_size 32 `
    --steps 8000 --eval_steps 400 --eval_freq 800 `
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth `
    --decoder_path ../ckpts/hidden_mnist.pth `
    --attack_layer hidden `
    --loss_i watson-vgg --loss_w bce --loss_c none --loss_d none `
    --lambda_i 1 --lambda_w 1 --lambda_c 0 --lambda_d 0
```

- With discriminator
```cmd
python finetune_dcgan.py mnist_critic --num_keys 1 `
    --num_bits 32 --img_size 28 --img_channels 1 --batch_size 32 `
    --steps 8000 --eval_steps 400 --eval_freq 800 `
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth `
    --discriminator_ckpt ../ckpts/dcgan_discriminator_mnist.pth `
    --decoder_path ../ckpts/hidden_mnist.pth `
    --attack_layer hidden `
    --loss_i watson-vgg --loss_w bce --loss_c bce --loss_d none `
    --lambda_i 1 --lambda_w 1 --lambda_c 0.1 --lambda_d 0
```

-- Adversarial scheme:
```cmd
python finetune_dcgan_adv.py mnist_adv --dataset MNIST `
    --num_keys 1 --num_bits 32 --img_size 28 --img_channels 1 --batch_size 32 `
    --steps 8000 --eval_steps 400 --eval_freq 800 `
    --generator_ckpt ../ckpts/dcgan_generator_mnist.pth `
    --discriminator_ckpt ../ckpts/dcgan_discriminator_mnist.pth `
    --decoder_path ../ckpts/hidden_mnist.pth `
    --attack_layer none `
    --loss_w bce --loss_c bce `
    --lambda_w 1 --lambda_c 1
```

###### on CIFAR-10
All experiments use R3GAN (a conditional GAN).

- Baseline
```cmd
python finetune_r3gan.py r3gan_cifar10_baseline --num_keys 1 \
    --num_bits 48 --img_size 32 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --decoder_path ../ckpts/hidden_replicate.pth \
    --attack_layer none \
    --loss_i watson-vgg --loss_w bce --loss_d none \
    --lambda_i 1 --lambda_w 0.1 --lambda_d 0
```

- With critic loss:
```cmd
python finetune_r3gan_critic.py r3gan_cifar10_critic --num_keys 1 \
    --num_bits 48 --img_size 32 --batch_size 16 \
    --steps 3000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --decoder_path ../ckpts/hidden_unet.pth \
    --attack_layer none \
    --loss_c bce --loss_w bce --loss_d none \
    --lambda_c 1 --lambda_w 0.6 --lambda_d 0
```

- With attack layer
```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --num_bits 48 --img_size 32 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer hidden \
    --loss_i watson-vgg --loss_w bce --loss_d none \
    --lambda_i 1 --lambda_w 0.1 --lambda_d 0
```

- With logit loss
```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --num_bits 48 --img_size 32 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer none \
    --loss_i watson-vgg --loss_w bce --loss_d kl \
    --lambda_i 1 --lambda_w 0.1 --lambda_d 0.1
```

- With both modifications
```cmd
python finetune_r3gan.py r3gan_cifar101 --num_keys 1 \
    --num_bits 48 --img_size 32 --batch_size 128 \
    --steps 2000 --eval_steps 100 \
    --generator_ckpt ../ckpts/r3gan_cifar10.pkl \
    --decoder_path ../ckpts/hidden_replicate.pth \
    --clf_ckpt ../ckpts/clf/resnet18_cifar10.pth \
    --attack_layer hidden \
    --loss_i watson-vgg --loss_w bce --loss_d kl \
    --lambda_i 1 --lambda_w 0.1 --lambda_d 0.1
```

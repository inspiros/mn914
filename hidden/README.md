Revisited HiDDeN
------

This code is inherited from
[HiDDeN implemented in Stable Signature repository](https://github.com/facebookresearch/stable_signature/tree/main/hidden),
please have a look at their `README.md`.
It aims to train image watermarking networks, consisting of a watermark encoder $W_E$ and a watermark decoder $W_D$.

In this project, we proposed an AE/U-Net-based watermark encoder architecture.
Also, we have reorganized and modified the code quite a lot to accommodate training on the grayscale datasets (MNIST).

<p align="center">
<img src="../resources/hidden_vs_unet.png" height="320"/>
</p>

## Module Structure

```
hidden/
│   README.md
│   requirements.txt
└───data/                <== (optional) Dataset
└───models/              <== Contains encoders, decoders, attack_layers, attenuations
└───notebooks/           <== Demo notebooks
└───ops/                 <== Image operators (attacks & metrics)
│   └───attacks/
│   └───metrics/
│   └───utils.py
└───outputs/             <== (optional) All outputs (checkpoints and logs)
│   loss.py
│   main.py              <-- Entry point for training
│   transforms.py
│   utils.py
```

## Usage

### Training

The main script is `main.py`.
These commands reproduce our experiments:

#### Experiments on MNIST

- HiDDeN $W_E$:
```cmd
python main.py mnist \
    --dataset MNIST --img_size 28 --img_channels 1 \
    --num_bits 32 --batch_size 16 --epochs 100 --eval_freq 5 \
    --encoder hidden --generate_delta True --decoder hidden \
    --optimizer Lamb,lr=2e-2 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w bce --loss_i mse --loss_p watson-vgg \
    --lambda_w 1 --lambda_i 0.001 --lambda_p 0.01
```
- U-Net-based $W_E$ _(with $W_D$ fine-tuned from the above experiment)_:
```cmd
python main.py mnist_unet \
    --dataset MNIST --img_size 28 --img_channels 1 \
    --num_bits 32 --batch_size 16 --epochs 100 --eval_freq 5 \
    --encoder unet --generate_delta False --decoder hidden \
    --resume_from ../ckpts/hidden_mnist.pth \
    --optimizer Lamb,lr=2e-2 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5 \
    --loss_w bce --loss_i mse --loss_p watson-vgg \
    --lambda_w 1 --lambda_i 0.02 --lambda_p 0.8
```

#### Experiments on COCO

Since training from scratch on small RGB dataset such as CIFAR-10 is extremely hard to converge,
we instead use the **MS COCO** resized to $128\times128$.
- HiDDeN $W_E$: We use the `hidden_replicate.pth` weights provided by the authors.
- U-Net-based $W_E$ _(with $W_D$ fine-tuned from `hidden_replicate.pth`)_:
```cmd
python main.py coco_unet \
    --train_dir ../data/coco/images/train2014/ --val_dir ../data/coco/images/val2014/ \
    --img_size 128 --img_channels 1 \
    --num_bits 32 --batch_size 16 --epochs 30 --eval_freq 5 \
    --encoder unet --generate_delta False --decoder hidden \
    --resume_from ../ckpts/hidden_replicate.pth \
    --optimizer Lamb,lr=2e-2 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=30,warmup_lr_init=1e-6,warmup_t=5 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w bce --loss_i mse --loss_p watson-vgg \
    --lambda_w 1 --lambda_i 0.02 --lambda_p 0.8
```

### Evaluating

Run the `eval.py` script with almost identical parameters as the `main.py` to evaluate the saved checkpoints.
The evaluation procedure is split to a separate file as we want to implement a comprehensive evaluation on many
different image-level attacks.


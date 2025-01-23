# Image Watermarking with Revisited HiDDeN

This code is inherited from
[HiDDeN implemented in Stable Signature repository](https://github.com/facebookresearch/stable_signature/tree/main/hidden),
please have a look at their `README.md`.
We have reorganized and modified it quite a lot to accommodate training on the grayscale datasets (MNIST).

## Setup

### Requirements

See [`requirements.txt`](requirements.txt).

```cmd
pip install -r requirements.txt
```

### Data

Instead of the [COCO](https://cocodataset.org/) dataset, we are going to use the MNIST dataset for
computational frugality.
Fortunately, we can rely on `torchvision.datasets`.

## Module Structure

```
hidden/
│   README.md
│   requirements.txt
└───ckpt/                <== (optional) Final checkpoints
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

The main script is in `main.py`. It can be used to train the watermark encoder and decoder networks.

To run it on one GPU, use the following command:

```bash
torchrun --nproc_per_node=1 main.py --dist False
```

To run it on multiple GPUs, use the following command:

```bash
torchrun --nproc_per_node=$GPUS$ main.py --local_rank 0
```

#### Options

<details>
<summary><span style="font-weight: normal;">Experiment Parameters</span></summary>

- `--dataset`: Named dataset in `torchvision.datasets`. Default: None
- `--data_dir`: Path to the directory to download the dataset if `--dataset` is defined. Default: "data"
- `--train_dir`: Path to the directory containing the training data. Default: "data/coco/train"
- `--val_dir`: Path to the directory containing the validation data. Default: "data/coco/val"
- `--output_dir`: Output directory for logs and images. Default: "outputs"
- `--data_mean`: Data mean for normalization. Default: None (inferred)
- `--data_std`: Data std for normalization. Default: None (inferred)
- `--verbose`: Verbosity level for output during training. Default: 1
- `--seed`: Random seed. Default: 0

</details>

<details>
<summary><span style="font-weight: normal;">Marking Parameters</span></summary>

- `--num_bits`: Number of bits in the watermark. Default: 32
- `--redundancy`: Redundancy of the watermark in the decoder (the output is bit is the sum of redundancy bits). Default:
  1
- `--img_size`: Image size during training. Having a fixed image size during training improves efficiency thanks to
  batching. The network can generalize (to a certain extent) to arbitrary resolution at test time. Default: 128
- `--img_channels`: Image channels. Default: None (inferred)

</details>

<details>
<summary><span style="font-weight: normal;">Encoder Parameters</span></summary>

- `--encoder`: Encoder type (e.g., "hidden", "dvmark", "vit"). Default: "hidden"
- `--encoder_depth`: Number of blocks in the encoder. Default: 4
- `--encoder_channels`: Number of channels in the encoder. Default: 64
- `--use_tanh`: Use tanh scaling. Default: True

</details>

<details>
<summary><span style="font-weight: normal;">Decoder Parameters</span></summary>

- `--decoder`: Decoder type (e.g., "hidden"). Default: "hidden"
- `--decoder_depth`: Number of blocks in the decoder. Default: 8
- `--decoder_channels`: Number of channels in the decoder. Default: 64

</details>

<details>
<summary><span style="font-weight: normal;">Training Parameters</span></summary>

- `--bn_momentum`: Momentum of the batch normalization layer. Default: 0.01
- `--eval_freq`: Frequency of evaluation during training (in epochs). Default: 1
- `--saveckp_freq`: Frequency of saving checkpoints (in epochs). Default: 100
- `--saveimg_freq`: Frequency of saving images (in epochs). Default: 10
- `--resume_from`: Checkpoint path to resume training from.
- `--scaling_w`: Scaling of the watermark signal. Default: 1.0
- `--scaling_i`: Scaling of the original image. Default: 1.0

</details>

<details>
<summary><span style="font-weight: normal;">Optimization Parameters</span></summary>

- `--epochs`: Number of epochs for optimization. Default: 400
- `--optimizer`: Optimizer to use (e.g., "Adam"). Default: "Adam"
- `--scheduler`: Learning rate scheduler to use (ex: "
  CosineLRScheduler,lr_min=1e-6,t_initial=400,warmup_lr_init=1e-6,warmup_t=5"). Default: None
- `--lambda_w`: Weight of the watermark loss. Default: 1.0
- `--lambda_i`: Weight of the image loss. Default: 0.0
- `--loss_margin`: Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. Default: 1.0
- `--loss_w_type`: Loss type for watermark loss ("bce" or "cossim"). Default: 'bce'
- `--loss_i_type`: Loss type for image loss ("mse" or "l1"). Default: 'mse'

</details>

<details>
<summary><span style="font-weight: normal;">Loader Parameters</span></summary>

- `--batch_size`: Batch size for training. Default: 16
- `--batch_size_eval`: Batch size for evaluation. Default: 64
- `--workers`: Number of workers for data loading. Default: 8

</details>

<details>
<summary><span style="font-weight: normal;">Attenuation Parameters</span></summary>

Additonally, the codebase allows to train with a just noticeable difference map (JND) to attenuate the watermark signal
in the perceptually sensitive regions of the image.
This can also be added at test time only, at the cost of some accuracy.

- `--attenuation`: Attenuation type. Default: None
- `--scale_channels`: Use channel scaling. Default: False

</details>

<details>
<summary><span style="font-weight: normal;">Data Augmentation Parameters</span></summary>

- `--data_augmentation`: Type of data augmentation to use at marking time ("combined", "kornia", "none"). Default: "
  combined"
- `--p_crop`: Probability of the crop augmentation. Default: 0.5
- `--p_res`: Probability of the resize augmentation. Default: 0.5
- `--p_blur`: Probability of the blur augmentation. Default: 0.5
- `--p_jpeg`: Probability of the JPEG compression augmentation. Default: 0.5
- `--p_rot`: Probability of the rotation augmentation. Default: 0.5
- `--p_color_jitter`: Probability of the color jitter augmentation. Default: 0.5

</details>

<details>
<summary><span style="font-weight: normal;">Distributed Training Parameters</span></summary>

- `--debug_slurm`: Enable debugging for SLURM.
- `--local_rank`: Local rank for distributed training. Default: -1
- `--master_port`: Port for the master process. Default: -1
- `--dist`: Enable distributed training. Default: False
- `--device`: Device to train on if distributed training is disabled. Default: cuda:0

</details>

#### Example

The following command resembles the one that reproduces the results in the original paper, but for our own settings:

- On Linux:
```cmd
python main.py \
  --dataset MNIST --eval_freq 5 \
  --img_size 28 --img_channels 1 --num_bits 16 --batch_size 128 --epochs 100 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 \
  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1
```
- On Windows:
```cmd
python main.py `
  --dataset MNIST --eval_freq 5 `
  --img_size 28 --img_channels 1 --num_bits 16 --batch_size 128 --epochs 100 `
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 `
  --optimizer Lamb,lr=2e-2 `
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 `
  --scaling_w 0.3 --scale_channels False --attenuation none `
  --loss_w_type bce --loss_margin 1
```

To enable distributed training, use `torchrun --nproc_per_node=[procs] main.py` instead of `python main.py` and
define `--dist`.

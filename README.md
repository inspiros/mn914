MN914 Joint Project - Towards Semantic Signature
------

This repo is intended for storing our code for the joint project.
It tries to apply Stable Signature for GAN.

## Setup

### Requirements

See [`requirements.txt`](requirements.txt).

```cmd
pip install -r requirements.txt
```

### Download Pre-trained Weights

Run the following script to download all necessary checkpoints to a folder named `ckpts`.

```cmd
python tools/download_weights.py
```

###### Recommendations

It is highly recommended to create a virtual environment in the project root. For example:

```cmd
python -m venv .venv
```

Then activate it with the following command:

- For Windows:
```cmd
.\.venv\Scripts\activate.ps1
```
- For Linux:
```cmd
./.venv/Scripts/activate.bat
```

## Project Structure

```
mn914/
│   README.md
│   requirements.txt
└───.venv/               <== (optional) virtual environment, you have to create it yourself
└───ckpts/               <== Final model checkpoints
└───hidden/              <== HiDDeN submodule
└───stable_signature/    <== Stable Signature submodule
│   └───models/
│       └───dcgan/       <-- DCGAN
│       └───r3gan/       <-- R3GAN
│       └───resnet/      <-- ResNet classifiers
└───resources/           <== Where we store resources (figures, etc.)
```

## Development

Notes for developers.

#### How to purge file from history

```cmd
git filter-branch -f --tree-filter 'rm -f <path_to_file>' HEAD
git push origin --force --all
```

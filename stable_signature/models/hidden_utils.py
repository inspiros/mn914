# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch

from hidden.models import HiddenEncoder, HiddenDecoder, resnet50_decoder


def get_hidden_encoder(num_bits, num_blocks=4, channels=64, in_channels=3):
    encoder = HiddenEncoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels, in_channels=in_channels)
    return encoder


def get_hidden_encoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    encoder_ckpt = {k.replace('module.', '').replace('encoder.', ''): v for k, v in ckpt['encoder_decoder'].items()
                    if 'encoder' in k}
    return encoder_ckpt


def get_hidden_decoder(decoder, num_bits, num_blocks=7, channels=64, in_channels=3):
    if decoder == 'hidden':
        return HiddenDecoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels, in_channels=in_channels)
    elif decoder == 'resnet':
        return resnet50_decoder(num_bits, in_channels, low_resolution=True)
    else:
        raise ValueError('Unknown decoder type')


def get_hidden_decoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    decoder_ckpt = {k.replace('module.', '').replace('decoder.', ''): v for k, v in ckpt['encoder_decoder'].items()
                    if 'decoder' in k}
    return decoder_ckpt


def instantiate_from_config(config):
    if not 'target' in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError('Expected key "target" to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt, device=None, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, weights_only=False, map_location=device)
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd['global_step']}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model).to(device)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)
    model.eval()
    return model

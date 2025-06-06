import numpy as np
import torch.nn as nn


class ShiftWrapper(nn.Module):
    r"""
    Extension for 2-dimensional inout loss functions. 
    Shifts the inputs by up to 4 pixels. Uses replication padding.

    Parameters:
        lossclass: class of the individual loss functions
        trainable: bool, if True parameters of the loss are trained.
        args: tuple, arguments for instantiation of loss fun
        kwargs: dict, key word arguments for instantiation of loss fun
    """

    def __init__(self, lossclass, args, kwargs):
        super().__init__()

        # submodules
        self.add_module('loss', lossclass(*args, **kwargs))

        # shift amount
        self.max_shift = 8

        # padding
        self.pad = nn.ReplicationPad2d(self.max_shift // 2)

    def forward(self, input, target):
        # convert color space
        input = self.pad(input)
        target = self.pad(target)

        shift_x = np.random.randint(self.max_shift)
        shift_y = np.random.randint(self.max_shift)

        input = input[:, :, shift_x:-(self.max_shift - shift_x), shift_y:-(self.max_shift - shift_y)]
        target = target[:, :, shift_x:-(self.max_shift - shift_x), shift_y:-(self.max_shift - shift_y)]

        return self.loss(input, target)

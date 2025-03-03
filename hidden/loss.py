"""
Notes: Deprecated in favor of stable signature losses
"""
import torch
import torch.nn.functional as F

__all__ = ['message_loss', 'image_loss']


def message_loss(fts, targets, m, loss_type='mse'):
    r"""
    Compute the message loss.

    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
        loss_type: the type of loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts / m), 0.5 * (targets + 1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')


def image_loss(imgs, imgs_ori, loss_type='mse'):
    r"""
    Compute the image loss.

    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

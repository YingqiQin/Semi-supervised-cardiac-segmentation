import numpy as np
import cv2
import sys
# from itertools import combinations, permutations
import random
from scipy import ndimage
from numpy.linalg import inv
import math


def crop_3D_image(image, cx, cy, cz, size, constant_values=0):
    X, Y, Z = image.shape[:3]
    rX = size[0] // 2
    rY = size[1] // 2
    rZ = size[2] // 2
    x1, x2 = cx - rX, cx + (size[0] - rX)
    y1, y2 = cy - rY, cy + (size[1] - rY)
    z1, z2 = cz - rZ, cz + (size[2] - rZ)
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def crop_image(image, cx, cy, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    rX = size[0] // 2
    rY = size[1] // 2
    x1, x2 = cx - rX, cx + (size[0] - rX)
    y1, y2 = cy - rY, cy + (size[1] - rY)
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def augment_data(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
    aug_image = np.zeros_like(image)
    aug_label = np.zeros_like(label)

    for i in range(image.shape[-1]):
        tem_image = image[:, :, i]
        tem_label = label[:, :, i]

        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [int(shift * np.random.uniform(-1, 1)),
                     int(shift * np.random.uniform(-1, 1))]
        rotate_val = rotate * np.random.uniform(-1, 1)
        scale_val = 1 + scale * np.random.uniform(-1, 1)

        tem_image = tem_image * (1 + intensity * np.random.uniform(-1, 1))

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = tem_image.shape
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        M = np.concatenate((M, np.array([[0, 0, 1]])), axis=0)

        aug_image[:, :, i] = ndimage.interpolation.affine_transform(tem_image, inv(M), order=1)
        aug_label[:, :, i] = ndimage.interpolation.affine_transform(tem_label, inv(M), order=0)

    return aug_image, aug_label


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.6f,' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


if __name__ == '__main__':
    import torch

    x = torch.zeros(112, 112, 4)
    pred = crop_image(x, 56, 56, (250, 250), constant_values=0)
    print(pred.shape)

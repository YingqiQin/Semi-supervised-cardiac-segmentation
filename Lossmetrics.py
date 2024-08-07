import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np


def DSC_avrage(input, target):
    input = F.softmax(input, dim=1)
    # input = input[:,1:,:,:]
    # target = target[:, 1:, :, :]
    dsc = 0.0
    smooth = 1e-5
    for ind in range(1, input.shape[1]):
        input_flatten = input[:, ind, :, :].flatten()
        target_flatten = target[:, ind, :, :].flatten()
        intersection = torch.sum(input_flatten * target_flatten)
        dsc = dsc + (2. * intersection + smooth) / (torch.sum(input_flatten) + torch.sum(target_flatten) + smooth)
    dsc = dsc / (input.shape[1] - 1)
    return dsc


def DSC_average(input, target):
    num_class = input.shape[1]
    input = F.softmax(input, dim=1)
    input = torch.argmax(input, dim=1)
    # target = torch.argmax(target,dim=1)
    dsc = torch.zeros(num_class - 1, dtype=torch.float32)
    smooth = 1e-5
    for i in range(1, num_class):
        A = (input == i).to(torch.float32)
        B = (target == i).to(torch.float32)
        intersection = torch.sum(A * B)
        dsc[i - 1] = (2. * intersection) / (torch.sum(A) + torch.sum(B) + smooth)
    dsc = torch.mean(dsc)
    return dsc


def DSC_3D_average(input, target):
    num_class = input.shape[0]
    dsc = torch.zeros(num_class - 1, dtype=torch.float32)
    smooth = 1e-5
    input = torch.argmax(input, dim=0)
    target = torch.argmax(target, dim=0)
    for i in range(1, num_class):
        A = (input == i).to(torch.float32)
        B = (target == i).to(torch.float32)
        intersection = torch.sum(A * B)
        dsc[i - 1] = (2. * intersection + smooth) / (torch.sum(A) + torch.sum(B) + smooth)
    # input_flatten = input.flatten()
    # target_flatten = target.flatten()
    # intersection = torch.sum(input_flatten * target_flatten)
    # dsc+= (2. * intersection + smooth) / (torch.sum(input_flatten) + torch.sum(target_flatten) + smooth)
    dsc = torch.mean(dsc)
    return dsc


def DSC_3D(input, target):
    num_class = input.shape[0]
    dsc = torch.zeros(num_class - 1, dtype=torch.float32)
    smooth = 1e-5
    input = torch.argmax(input, dim=0)
    target = torch.argmax(target, dim=0)
    for i in range(1, num_class):
        A = (input == i).to(torch.float32)
        B = (target == i).to(torch.float32)
        intersection = torch.sum(A * B)
        dsc[i - 1] = (2. * intersection + smooth) / (torch.sum(A) + torch.sum(B) + smooth)
    # for i in range(3):
    #     dsc = 0.0
    #     input_flatten = input[i,:,:,:].flatten()
    #     target_flatten = target[i,:,:,:].flatten()
    #     intersection = torch.sum(input_flatten * target_flatten)
    #     dsc += (2. * intersection + smooth) / (torch.sum(input_flatten) + torch.sum(target_flatten) + smooth)
    #     dsc_classes.append(dsc)
    return dsc[0], dsc[1], dsc[2]

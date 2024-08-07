import scipy.misc
import csv
import os
import random
import numpy as np
import nibabel as nib
import shutil
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import image_utils
from scipy import ndimage
from numpy.linalg import inv
from matplotlib import pyplot as plt
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import cv2


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def get_image_list(csv_file):
    image_list, label_list = [], []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_list.append(row['image_filenames'].replace('\t', '').strip())
            label_list.append(row['label_filenames'].replace('\t', '').strip())

    data_list = {}
    data_list['image_filenames'] = image_list
    data_list['label_filenames'] = label_list

    return data_list


def norm_crop_aug_shuffle_data(image, label, image_size):
    clip_min = np.percentile(image, 1)
    clip_max = np.percentile(image, 99)
    image = np.clip(image, clip_min, clip_max)
    image = (image - image.min()) / float(image.max() - image.min())
    image = image.squeeze()
    label = label.squeeze()
    x, y, z = image.shape
    x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
    image = image_utils.crop_3D_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)
    label = image_utils.crop_3D_image(label, x_centre, y_centre, z_center, image_size, constant_values=0)
    return image, label


def getperdata(data_dir, data_list, index):
    image_name = data_dir + '/' + data_list['image_filenames'][index]
    label_name = data_dir + '/' + data_list['label_filenames'][index]

    nib_image = nib.load(image_name)
    nib_label = nib.load(label_name)

    whole_image = nib_image.get_data().squeeze()
    whole_label = nib_label.get_data().squeeze()

    whole_image = whole_image.astype('float32')
    whole_label = whole_label.astype('int8')


    return whole_image, whole_label


def load_original_data(data_dir, data_csv, augment=False, shuffle=False, image_size=(256, 256, 18)):
    imagelist = []
    labellist = []
    Datafilelist = get_image_list(data_csv)

    for index in range(len(Datafilelist['image_filenames'])):
        # print('Get data from object %d ' % (index))
        # get a volume of one subject
        whole_image, whole_label = getperdata(data_dir, Datafilelist, index)
        image, label = norm_crop_aug_shuffle_data(whole_image, whole_label, image_size)
        imagelist.append(image)
        labellist.append(label)
    return imagelist, labellist


def get_batch_valid(data_dir, data_csv, image_size):
    imagelist = []
    labellist = []
    Datafilelist = get_image_list(data_csv)

    for index in range(len(Datafilelist['image_filenames'])):
        # print('Get data from object %d ' % (index))
        # get a volume of one subject
        whole_image, whole_label = getperdata(data_dir, Datafilelist, index)
        image, label = norm_crop_aug_shuffle_data(whole_image, whole_label, image_size)
        tmp_image_list = []
        tmp_label_list = []
        for i in range(image.shape[2]):
            tem_image = np.expand_dims(image[:, :, i], axis=0)
            tmp_image_list.append(tem_image)
            tmplabel = np.expand_dims(label[:, :, i], axis=0).astype('int8')
            tmp_label_list.append(tmplabel)
        imagelist.append(np.concatenate(tmp_image_list, axis=0))
        labellist.append(np.concatenate(tmp_label_list, axis=0))
    return imagelist, labellist


def get_batch(images_list, labels_list, batch_size):
    rand_subj_indices = [i for i in range(len(images_list))]
    random.shuffle(rand_subj_indices)
    batch_image_array = []
    batch_label_array = []
    tmp_image = []
    tmp_label = []
    for j in range(len(images_list)):
        if (j + 1) % (batch_size) == 0:
            tmp_image.append(images_list[rand_subj_indices[j]])
            tmp_label.append(labels_list[rand_subj_indices[j]])

            batch_image_array.append(np.concatenate(tmp_image, axis=0))
            batch_label_array.append(np.concatenate(tmp_label, axis=0))
            tmp_image = []
            tmp_label = []
        else:
            tmp_image.append(images_list[rand_subj_indices[j]])
            tmp_label.append(labels_list[rand_subj_indices[j]])
    if (j + 1) % (batch_size) != 0:
        batch_image_array.append(np.concatenate(tmp_image, axis=0))
        batch_label_array.append(np.concatenate(tmp_label, axis=0))
    return batch_image_array, batch_label_array


def augment_data_2D(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
    aug_image = np.zeros_like(image)
    aug_label = np.zeros_like(label)

    tem_image = image[:, :]
    tem_label = label[:, :]

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

    aug_image[:, :] = ndimage.interpolation.affine_transform(tem_image, inv(M), order=1)
    aug_label[:, :] = ndimage.interpolation.affine_transform(tem_label, inv(M), order=0)

    return aug_image, aug_label


def augment_data_batch(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
    aug_image = np.zeros_like(image)
    aug_label = np.zeros_like(label)
    for i in range(image.shape[0]):
        tem_image = image[i, :, :]
        tem_label = label[i, :, :]

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

        aug_image[i, :, :] = ndimage.interpolation.affine_transform(tem_image, inv(M), order=1)
        aug_label[i, :, :] = ndimage.interpolation.affine_transform(tem_label, inv(M), order=0)

    return aug_image, aug_label


def crop_batch_data(image, size, shift_value, constant_values=0):
    image = np.transpose(image, [1, 2, 0])
    X, Y = image.shape[:2]
    shift_val = [shift_value, shift_value]
    cx = X // 2 + shift_val[0]
    cy = Y // 2 + shift_val[1]
    rX = size[0] // 2
    rY = size[1] // 2
    x1, x2 = cx - rX, cx + (size[0] - rX)
    y1, y2 = cy - rY, cy + (size[1] - rY)
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)

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
    crop = np.transpose(crop, [2, 0, 1])
    return crop


def onehot(images, numclasses):
    b, x, y = images.shape
    ont_hot_image = np.zeros([b, numclasses, x, y], dtype='int8')
    for i in range(b):
        for j in range(numclasses):
            j_matrix = (images[i, :, :] == j) * np.ones_like(images[i, :, :])
            ont_hot_image[i, j, :, :] = j_matrix
    return ont_hot_image


class categorical_cross_entropy(nn.Module):
    def __init__(self):
        super(categorical_cross_entropy, self).__init__()

    def forward(self, input, label):
        batchsz = input.shape[0]
        channel = input.shape[1]
        h, w = input.size(2), input.size(3)
        label_sum = torch.sum(label, dim=[0, 2, 3])
        p_i = F.softmax(input, dim=1)
        loss = label * torch.log(p_i + 1e-7)
        loss = -loss.flatten().sum()
        loss /= (batchsz * h * w)
        return loss


def gassian_operation(img):
    # img:b,h,w
    for i in range(img.shape[0]):
        aug = np.random.randint(0, 4, [1])
        if aug == 0:
            std = np.random.uniform(0.05, 0.1)

            img[i, :, :] = img[i, :, :] + np.random.normal(0, scale=std, size=img[i].shape)
        if aug == 1:
            std = np.random.uniform(0.25, 1.5)

            img[i, :, :] = gaussian_filter(img[i, :, :], std, order=0)
        elif aug == 2:
            alpha = np.random.uniform(5, 15)
            std = np.random.uniform(0.25, 1.5)

            i_blurred = gaussian_filter(img[i, :, :], std, order=0)
            i_filterblurred = gaussian_filter(i_blurred, std, order=0)
            img[i, :, :] = i_blurred + alpha * (i_blurred - i_filterblurred)
    return img


def gamma_correction(img):
    # img : b,h,w
    Image = img.copy()
    for i in range(img.shape[0]):
        gamma = np.random.uniform(0.5, 1.5, [1])
        Image[i, :, :] = img[i, :, :] ** gamma
    return Image

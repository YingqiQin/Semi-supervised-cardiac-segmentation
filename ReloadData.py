import csv
from abc import ABC
import cv2
import numpy as np
import torch
from torch.utils import data
import nibabel as nib
from scipy import ndimage
from numpy.linalg import inv
import torch.nn.functional as F


class WholeDataset(data.Dataset, ABC):
    def __init__(self, data_dir, sup_data_csv, unsup_data_csv, image_size=(256, 256, 18), mode='train'):
        super(WholeDataset, self).__init__()
        self.data_dir = data_dir
        self.sup_data_csv = sup_data_csv
        self.unsup_data_csv = unsup_data_csv
        self.image_size = image_size
        self.mode = mode
        self.sup_img_file_list, self.sup_label_file_list = self.file2list(self.sup_data_csv)
        self.unsup_img_file_list, self.labeled_frames = self.unsup_file2list(self.unsup_data_csv)
        self.sup_img_list = []
        self.sup_label_list = []
        self.unsup_img_list = []

        for index in range(len(self.sup_img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.sup_img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.sup_label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size)
            label = self.label_preprocessing(whole_label, self.image_size)

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.sup_img_list.append(temp_image)

            temp_label = np.expand_dims(label[:, :, :], axis=0)
            self.sup_label_list.append(temp_label)

        for index in range(len(self.unsup_img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.unsup_img_file_list, index)
            whole_img = np.delete(whole_img, self.labeled_frames[index] - 1, axis=-1)
            _t = whole_img.shape[-1]
            _image = [self.data_preprocessing(whole_img[:, :, :, i], image_size=self.image_size) for i in
                      range(_t)]
            temp_image = np.stack(_image, axis=-1)

            self.unsup_img_list.append(temp_image)

    def __getitem__(self, item):

        sup_img = self.sup_img_list[item]
        sup_label = self.sup_label_list[item]

        sup_image = torch.Tensor(sup_img)
        sup_target = torch.LongTensor(sup_label)

        unsup_img = self.unsup_img_list[item]

        unsup_image = torch.Tensor(unsup_img)

        if self.mode == 'train':
            return sup_image, sup_target, unsup_image

        elif self.mode == 'valid':
            return sup_image, sup_target

    def __len__(self):
        return len(self.sup_img_list)

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        return whole_image

    def file2list(self, file_csv):
        img_list = []
        label_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                label_list.append(line['label_filenames'])

        return img_list, label_list

    def unsup_file2list(self, file_csv):
        img_list = []
        label_list = []
        labeled_frame = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                labeled_frame.append(int(line['frames']))

        return img_list, labeled_frame

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
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

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label


class PretrainedRegistrationDataset(data.Dataset, ABC):
    def __init__(self, data_dir, ED_data_csv, ES_data_csv, image_size=(224, 224, 18), mode='train'):
        self.data_dir = data_dir
        self.ED_data_csv = ED_data_csv
        self.ES_data_csv = ES_data_csv
        self.image_size = image_size
        self.mode = mode
        self.ED_img_file_list, self.ED_label_file_list = self.file2list(self.ED_data_csv)
        self.ES_img_file_list, self.ES_label_file_list = self.file2list(self.ES_data_csv)

        self.ED_img_list = []
        self.ED_label_list = []

        self.ES_img_list = []
        self.ES_label_list = []

        for index in range(len(self.ED_img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.ED_img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.ED_label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size)
            label = self.label_preprocessing(whole_label, self.image_size)

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.ED_img_list.append(temp_image)

            temp_label = np.expand_dims(label[:, :, :], axis=0)
            self.ED_label_list.append(temp_label)

        for index in range(len(self.ES_img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.ES_img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.ES_label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size)
            label = self.label_preprocessing(whole_label, self.image_size)

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.ES_img_list.append(temp_image)

            temp_label = np.expand_dims(label[:, :, :], axis=0)
            self.ES_label_list.append(temp_label)

    def __len__(self):
        return len(self.ED_img_list)

    def __getitem__(self, item):
        ED_img = self.ED_img_list[item]
        ED_label = self.ED_label_list[item]

        ED_image = torch.Tensor(ED_img)
        ED_target = torch.LongTensor(ED_label)

        ES_img = self.ES_img_list[item]
        ES_label = self.ES_label_list[item]

        ES_image = torch.Tensor(ES_img)
        ES_target = torch.LongTensor(ES_label)

        return ED_image, ED_target, ES_image, ES_target

    def file2list(self, file_csv):
        img_list = []
        label_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                label_list.append(line['label_filenames'])

        return img_list, label_list

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        return whole_image

    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
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

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label

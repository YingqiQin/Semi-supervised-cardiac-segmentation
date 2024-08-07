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
from collections import defaultdict


class SemiSegDataset(data.Dataset, ABC):
    def __init__(self, data_dir, unsup_data_csv, sup_data_csv, image_size=(224, 224, 18), mode='train'):
        super(SemiSegDataset, self).__init__()

        self.data_dir = data_dir
        self.ED_data_csv = unsup_data_csv
        self.ES_data_csv = sup_data_csv
        self.image_size = image_size
        self.mode = mode
        self.ED_img_file_list = self.file2list(self.ED_data_csv, no_label=True)
        self.ES_img_file_list, self.ES_label_file_list = self.file2list(self.ES_data_csv, no_label=False)

        self.ED_img_list = []

        self.ES_img_list = []
        self.ES_label_list = []

        for index in range(len(self.ED_img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.ED_img_file_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size)

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.ED_img_list.append(temp_image)

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
        if self.mode == 'train':

            ED_img = self.ED_img_list[item]
            ED_image = torch.Tensor(ED_img)

            ES_img = self.ES_img_list[item]
            ES_label = self.ES_label_list[item]

            ES_image = torch.Tensor(ES_img)
            ES_target = torch.LongTensor(ES_label)

            return ED_image, ES_image, ES_target
        elif self.mode == 'valid':
            ES_img = self.ES_img_list[item]
            ES_label = self.ES_label_list[item]

            ES_image = torch.Tensor(ES_img)
            ES_target = torch.LongTensor(ES_label)

            return ES_image, ES_target

    def file2list(self, file_csv, no_label=True):
        img_list = []
        label_list = []

        if not no_label:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])
                    label_list.append(line['label_filenames'])

            return img_list, label_list
        else:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])

            return img_list

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


class FullyDataSet(data.Dataset, ABC):
    def __init__(self, data_dir, data_csv, image_size=(224, 224), mode='train'):
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.image_size = image_size
        self.mode = mode
        self.img_file_list, self.label_file_list = self.file2list(self.data_csv)
        self.img_list = []
        self.label_list = []

        for index in range(len(self.img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size)
            label = self.label_preprocessing(whole_label, self.image_size)

            for i in range(image.shape[2]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)

                self.img_list.append(temp_image)
            for j in range(label.shape[2]):
                temp_label = np.expand_dims(label[:, :, j], axis=0)
                self.label_list.append(temp_label)

    def __getitem__(self, idx):

        if self.mode == 'train':
            img = self.img_list[idx]
            label = self.label_list[idx]
            aug_image, aug_label = self.augment_data_batch(img, label, shift=0, rotate=30, scale=0.2, intensity=0.2,
                                                           flip=False)
            shift_max = 10
            shift = int(shift_max * np.random.uniform(-1, 1))
            image = self.crop_batch_data(aug_image, self.image_size, shift_value=shift)
            target = self.crop_batch_data(aug_label, self.image_size, shift_value=shift).squeeze()

            image = torch.Tensor(image)
            target = torch.LongTensor(target)
        else:
            img = self.img_list[idx]
            label = self.label_list[idx].squeeze()

            image = torch.Tensor(img)
            target = torch.LongTensor(label)

        return image, target

    def __len__(self):
        return len(self.img_list)

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

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

    def crop_image(self, image, cx, cy, size, constant_values=0):
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

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label

    def augment_data_batch(self, image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
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

    def crop_batch_data(self, image, size, shift_value, constant_values=0):
        image = np.transpose(image, [1, 2, 0])
        X, Y = image.shape[:2]
        shift_val = [shift_value, shift_value]
        cx = X // 2 + shift_val[0] - 15
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


class RandomPairs(SemiSegDataset):
    def __getitem__(self, item):
        if self.mode == 'train':
            random_index = np.random.randint(low=0, high=len(self.ED_img_list))

            ED_img = self.ED_img_list[random_index]
            ED_image = torch.Tensor(ED_img)

            random_index = np.random.randint(low=0, high=len(self.ED_img_list))
            ES_img = self.ES_img_list[random_index]
            ES_label = self.ES_label_list[random_index]

            ES_image = torch.Tensor(ES_img)
            ES_target = torch.LongTensor(ES_label)

            return ED_image, ES_image, ES_target
        elif self.mode == 'valid':
            ES_img = self.ES_img_list[item % len(self.ES_img_list)]
            ES_label = self.ES_label_list[item % len(self.ES_img_list)]

            ES_image = torch.Tensor(ES_img)
            ES_target = torch.LongTensor(ES_label)

            return ES_image, ES_target

    def __len__(self):
        return len(self.ED_img_list) + len(self.ES_img_list)


class AllPhasesDataset(data.Dataset, ABC):
    def __init__(self, data_dir, all_data_csv, val_data_csv, image_size=(224, 224, 18), mode='train'):
        super(AllPhasesDataset, self).__init__()
        self.data_dir = data_dir
        self.data_csv = all_data_csv
        self.val_data_csv = val_data_csv
        self.image_size = image_size
        self.mode = mode
        self.img_file_list, self.label_file_list, self.label_phases_list = self.file2list(self.data_csv)
        self.val_img_file_list, self.val_label_file_list = self.file2list(self.val_data_csv, True)
        self.img_dict = []
        self.val_img_list = []
        self.val_label_list = []

        if self.mode == 'train':

            for index in range(len(self.img_file_list)):

                image_4d = self.getperdata(self.data_dir, self.img_file_list, index)
                whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
                image = self.data_preprocessing_4d(image_4d, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, self.image_size)
                dick = defaultdict(list)
                dick['labeled_image'].append(
                    np.expand_dims(image[:, :, :, int(self.label_phases_list[index]) - 1], axis=0))
                dick['label'].append(np.expand_dims(label, axis=0))
                for i in range(int(self.label_phases_list[index]) - 1):
                    dick['unlabeled_image'].append(np.expand_dims(image[:, :, :, i], axis=0))

                self.img_dict.append(dick)
                del dick
        else:
            for index in range(len(self.val_img_file_list)):
                image = self.getperdata(self.data_dir, self.val_img_file_list, index)
                label = self.getperdata(self.data_dir, self.val_label_file_list, index)
                image = self.data_preprocessing(image, image_size=image_size)
                label = self.label_preprocessing(label, self.image_size)
                self.val_img_list.append(np.expand_dims(image, axis=0))
                self.val_label_list.append(np.expand_dims(label, axis=0))

    def __len__(self):
        if self.mode == 'train':

            return len(self.img_dict)
        else:
            return len(self.val_img_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            labeled_img = self.img_dict[idx]['labeled_image'][0]
            label = self.img_dict[idx]['label'][0]
            unlabeled_img_list = [torch.Tensor(unlab_img) for unlab_img in self.img_dict[idx]['unlabeled_image']]
            labed_img = torch.Tensor(labeled_img)
            target = torch.LongTensor(label)
            return labed_img, target, unlabeled_img_list
        else:
            img = self.val_img_list[idx]
            label = self.val_label_list[idx]

            image = torch.Tensor(img)
            target = torch.LongTensor(label)

            return image, target

    def file2list(self, file_csv, val=False):
        img_list = []
        label_list = []
        label_phases = []
        if not val:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])
                    label_list.append(line['label_filenames'])
                    label_phases.append(line['label_phases'])

            return img_list, label_list, label_phases
        else:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])
                    label_list.append(line['label_filenames'])

            return img_list, label_list

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        return whole_image

    def data_preprocessing_4d(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z, _ = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

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


class AllPhasesDataset_Pretrained(data.Dataset, ABC):
    def __init__(self, data_dir, all_data_csv, val_data_csv, image_size=(224, 224, 18), mode='train'):
        super(AllPhasesDataset_Pretrained, self).__init__()
        self.data_dir = data_dir
        self.data_csv = all_data_csv
        self.val_data_csv = val_data_csv
        self.image_size = image_size
        self.mode = mode
        self.img_file_list, self.label_file_list, self.label_phases_list = self.file2list(self.data_csv)
        self.val_img_file_list, self.val_label_phases_list = self.file2list(self.val_data_csv, val=True)
        self.img_dict = []
        self.val_img_list = []
        self.val_label_list = []

        if self.mode == 'train':

            for index in range(len(self.img_file_list)):

                image_4d = self.getperdata(self.data_dir, self.img_file_list, index)
                whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
                image = self.data_preprocessing_4d(image_4d, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, self.image_size)
                dick = defaultdict(list)
                dick['labeled_image'].append(
                    np.expand_dims(image[:, :, :, int(self.label_phases_list[index]) - 1], axis=0))
                dick['label'].append(np.expand_dims(label, axis=0))
                for i in range(int(self.label_phases_list[index]) - 1):
                    dick['unlabeled_image'].append(np.expand_dims(image[:, :, :, i], axis=0))

                self.img_dict.append(dick)
                del dick
        else:
            for index in range(len(self.val_img_file_list)):
                image_4d = self.getperdata(self.data_dir, self.val_img_file_list, index)
                image = self.data_preprocessing_4d(image_4d, image_size=self.image_size)
                dick = defaultdict(list)
                dick['labeled_image'].append(
                    np.expand_dims(image[:, :, :, int(self.val_label_phases_list[index]) - 1], axis=0)
                )
                for i in range(int(self.val_label_phases_list[index]) - 1):
                    dick['unlabeled_image'].append(np.expand_dims(image[:, :, :, i], axis=0))
                self.img_dict.append(dick)
                del dick

    def __len__(self):

        return len(self.img_dict)

    def __getitem__(self, idx):
        if self.mode == 'train':
            labeled_img = self.img_dict[idx]['labeled_image'][0]
            label = self.img_dict[idx]['label'][0]
            unlabeled_img_list = [torch.Tensor(unlab_img) for unlab_img in self.img_dict[idx]['unlabeled_image']]
            labed_img = torch.Tensor(labeled_img)
            target = torch.LongTensor(label)
            return labed_img, target, unlabeled_img_list
        else:
            labeled_img = self.img_dict[idx]['labeled_image'][0]
            unlabeled_img_list = [torch.Tensor(unlab_img) for unlab_img in self.img_dict[idx]['unlabeled_image']]
            return labeled_img, unlabeled_img_list

    def file2list(self, file_csv, val=False):
        img_list = []
        label_list = []
        label_phases = []
        if not val:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])
                    label_list.append(line['label_filenames'])
                    label_phases.append(line['label_phases'])

            return img_list, label_list, label_phases
        else:
            with open(file_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    img_list.append(line['image_filenames'])
                    label_phases.append(line['label_phases'])

            return img_list, label_phases

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        return whole_image

    def data_preprocessing_4d(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z, _ = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

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

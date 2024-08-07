import argparse
from SegModel import *
import torch
import os
import random
import numpy as np
import torch.optim as optim
from MyDataSet import FullyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from RegNet import *
from image_utils import LR_Scheduler, crop_image
import torch.nn.functional as F
from Lossmetrics import *
from Load_Data import get_image_list
import nibabel as nib
import segmentation_models_pytorch as smp

device = torch.device('cuda:0')


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training(args):
    seed_torch(args.seed)
    train_output_dir = args.train_output_dir
    model_dir = train_output_dir + '/model'
    training_graph = train_output_dir + '/graph'
    train_csv = train_output_dir + '/csv'
    batch_size = args.batch_size

    learning_rate = args.learning_rate
    epochs = int(args.epochs)
    image_size = args.image_size
    num_classes = args.num_classes

    mkdir(train_output_dir)
    mkdir(model_dir)
    mkdir(training_graph)
    mkdir(train_csv)

    train_dataset = FullyDataSet(args.train_data_dir, args.train_data_list,
                                 args.image_size, 'train')
    valid_dataset = FullyDataSet(args.valid_data_dir, args.valid_data_list,
                                 args.image_size, 'valid')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print('starting training!')

    writer = SummaryWriter(training_graph)

    save_path = os.path.join(model_dir, 'Unet.pth')

    # net = SingleUnet('resnet50', classes=num_classes, deep_stem=True).cuda()
    net = smp.UnetPlusPlus(
        encoder_name='resnet50',
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        in_channels=1,
        classes=num_classes
    ).cuda()

    params1 = [p for p in net.parameters() if p.requires_grad]

    optimizer = optim.Adam(params1, lr=learning_rate)
    best_valid_dsc = 0.0
    final_train_dsc = 0.0

    for epoch in range(epochs):

        avdsc_train = 0.0
        avloss_train = 0.0

        train_bar = tqdm(train_loader)

        net.train()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()

            outputs1 = net(images.to(device))

            loss = F.cross_entropy(outputs1, labels.to(device))
            dsc = DSC_average(outputs1, labels.to(device))

            loss.backward()
            optimizer.step()

            avloss_train += loss.item()
            avdsc_train += dsc.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} dsc:{:.3f}".format(epoch + 1, epochs, loss, dsc)

        avdsc_valid = 0.0
        avloss_valid = 0.0
        net.eval()

        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data

                outputs1 = net(val_images.to(device))

                loss = F.cross_entropy(outputs1, val_labels.to(device))
                dsc = DSC_average(outputs1, val_labels.to(device))

                avloss_valid += loss.item()
                avdsc_valid += dsc.item()

        avdsc_train /= len(train_loader)
        avloss_train /= len(train_loader)

        avdsc_valid /= len(valid_loader)
        avloss_valid /= len(valid_loader)
        print(
            "train epoch[{}/{}] Training average loss"
            ":{:.3f} DSC:{:.3f}, Validation average loss:{:.3f},Validation average DSC:{:.3f}".format(
                epoch + 1, epochs,
                avloss_train, avdsc_train,
                avloss_valid, avdsc_valid))
        if avdsc_valid > best_valid_dsc:
            best_valid_dsc = avdsc_valid
            final_train_dsc = avdsc_train
            print("model saved!")
            torch.save(net.state_dict(), save_path)

        writer.add_scalar("BigAug Train/Loss", avloss_train, epoch)
        writer.add_scalar("BigAug Train/DSC", avdsc_train, epoch)
        writer.add_scalar("BigAug Validation/Loss", avloss_valid, epoch)
        writer.add_scalar("BigAug Validation/DSC", avdsc_valid, epoch)

    print("Finished Training!")


def testing(args):
    training_epochs = args.epochs
    image_size = args.image_size
    num_classes = int(args.num_classes)

    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    test_csv = test_output_dir + '/csv'
    pred_dir = test_output_dir + '/predictions'
    mkdir(test_csv)
    mkdir(pred_dir)

    model_path = os.path.join(model_dir, 'Unet.pth')
    net = smp.UnetPlusPlus(
        encoder_name='resnet50',
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        in_channels=1,
        classes=num_classes
    ).cuda()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    mkdir(test_output_dir)
    seed_torch(int(args.seed))
    data_list = get_image_list(args.test_data_list)
    print("using {} images for testing.".format(len(data_list['image_filenames'])))

    with torch.no_grad():
        for index in range(len(data_list['image_filenames'])):
            gt_name = data_list['label_filenames'][index]
            nib_gt = nib.load(os.path.join(args.test_data_dir, gt_name))

            gt = nib_gt.get_fdata()

            img_name = data_list['image_filenames'][index]
            nib_img = nib.load(os.path.join(args.test_data_dir, img_name))

            img = np.squeeze(nib_img.get_data().astype('float32'))

            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            img = np.clip(img, clip_min, clip_max)
            img = (img - img.min()) / float(img.max() - img.min())
            x, y, z = img.shape
            x_centre, y_centre = int(x / 2), int(y / 2)
            img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)

            pred_res = torch.zeros(img.shape, dtype=torch.int8)
            for i in range(img.shape[2]):
                tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(dim=0).unsqueeze(dim=0)
                outputs = net(tmp_image.to(device))
                tmp_prob = F.softmax(outputs, dim=1)
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_start = img_name.rfind('/')
            loc_end = img_name.find('.')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = os.path.join(savedirname, img_name[loc_start + 1: loc_end] + '_Pred.nii.gz')

            nib.save(nii_pred, pred_name)

    print("Finished Testing!")


def extra_test(args):
    training_epochs = args.epochs
    image_size = args.image_size
    num_classes = int(args.num_classes)

    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    test_csv = test_output_dir + '/csv'
    pred_dir = test_output_dir + '/series_predictions'
    mkdir(test_csv)
    mkdir(pred_dir)

    model_path = os.path.join(model_dir, 'Unet.pth')
    net = smp.UnetPlusPlus(
        encoder_name='resnet50',
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        in_channels=1,
        classes=num_classes
    ).cuda()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    mkdir(test_output_dir)
    seed_torch(int(args.seed))
    data_list = get_image_list(args.test_data_list)
    print("using {} images for testing.".format(len(data_list['image_filenames'])))

    with torch.no_grad():
        for index in range(len(data_list['image_filenames'])):
            gt_name = data_list['label_filenames'][index]
            nib_gt = nib.load(os.path.join(args.test_data_dir, gt_name))

            gt = nib_gt.get_fdata()

            img_name = data_list['image_filenames'][index]
            nib_img = nib.load(os.path.join(args.test_data_dir, img_name))

            img = np.squeeze(nib_img.get_data().astype('float32'))

            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            img = np.clip(img, clip_min, clip_max)
            img = (img - img.min()) / float(img.max() - img.min())
            x, y, z = img.shape
            x_centre, y_centre = int(x / 2), int(y / 2)
            img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)

            pred_res = torch.zeros(img.shape, dtype=torch.int8)
            for i in range(img.shape[2]):
                tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(dim=0).unsqueeze(dim=0)
                outputs = net(tmp_image.to(device))
                tmp_prob = F.softmax(outputs, dim=1)
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_start = img_name.rfind('/')
            loc_end = img_name.find('.')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = os.path.join(savedirname, img_name[loc_start + 1: loc_end] + '_Pred.nii.gz')

            nib.save(nii_pred, pred_name)

    print("Finished Testing!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--valid_data_dir", default='./ACDC/all/converted')
    parser.add_argument("--test_data_dir", default='./ACDC/all/converted')
    parser.add_argument("--train_data_list", default=None)
    parser.add_argument("--valid_data_list", default=None)
    parser.add_argument("--test_data_list", default=None)
    parser.add_argument("--train_output_dir", type=str, default=None)
    parser.add_argument("--seed", default=666)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--mode", default='train')
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--image_size", default=(224, 224))
    parser.add_argument("--test_output_dir", default=None)
    args = parser.parse_args()

    mode = args.mode

    if mode == 'train':
        training(args)
        testing(args)
    if mode == 'test':
        testing(args)
    if mode == 'external':
        extra_test(args)
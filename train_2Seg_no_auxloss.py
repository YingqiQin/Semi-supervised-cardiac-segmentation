import argparse
from SegModel import *
import torch
import os
import random
import numpy as np
import torch.optim as optim
from MyDataSet import SemiSegDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from RegNet import *
from image_utils import LR_Scheduler, crop_image, crop_3D_image
import torch.nn.functional as F
from DenseCRFLoss import *
from Lossmetrics import *
from Load_Data import get_image_list, augment_data_batch, crop_batch_data
import nibabel as nib
import csv

device = torch.device('cuda:0')


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def compute_dice(input, target, classes):
    smooth = 1e-5
    A = (input == classes).to(torch.float32)
    B = (target == classes).to(torch.float32)
    intersection = torch.sum(A * B)
    dsc = (2. * intersection) / (torch.sum(A) + torch.sum(B) + smooth)
    return dsc


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

    reg_batch_size = args.reg_batch_size
    seg_batch_size = args.seg_batch_size

    learning_rate = args.learning_rate
    epochs = int(args.epochs)
    image_size = args.image_size
    num_classes = args.num_classes

    mkdir(train_output_dir)
    mkdir(model_dir)
    mkdir(training_graph)
    mkdir(train_csv)

    train_dataset = SemiSegDataset(data_dir=args.train_data_dir, unsup_data_csv=args.train_unsup_data_csv,
                                   sup_data_csv=args.train_sup_data_csv,
                                   image_size=args.image_size, mode='train')
    valid_dataset = SemiSegDataset(data_dir=args.train_data_dir, unsup_data_csv=args.valid_unsup_data_csv,
                                   sup_data_csv=args.valid_sup_data_csv,
                                   image_size=args.image_size, mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=reg_batch_size,
                              shuffle=True, num_workers=0)

    valid_loader = DataLoader(valid_dataset, batch_size=reg_batch_size,
                              shuffle=False, num_workers=0)
    print("Dataset Initialized!")

    writer = SummaryWriter(training_graph)

    reg_save_path = os.path.join(model_dir, 'Registration.pth')
    seg_save_path1 = os.path.join(model_dir, 'SegNet1.pth')
    seg_save_path2 = os.path.join(model_dir, 'SegNet2.pth')

    print("----Initial Registration Network----")

    RegModel = VMDiff(args.image_size, args.steps, args.downsize).cuda()
    stn = SpatialTransformer(args.image_size).cuda()
    if args.resume is not None:
        RegModel.load_state_dict(torch.load(args.resume, map_location=device))

    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    else:
        sim_loss_fn = ncc_loss
    grad_loss_fn = compute_gradient

    SegNet1 = SingleUnet('resnet50', 'imagenet', classes=num_classes, deep_stem=True).cuda()
    SegNet2 = SingleUnet('resnet50', 'imagenet', classes=num_classes, deep_stem=True).cuda()

    params_reg = [p for p in RegModel.parameters() if p.requires_grad]
    params_seg_1 = [p for p in SegNet1.parameters() if p.requires_grad]
    params_seg_2 = [p for p in SegNet2.parameters() if p.requires_grad]

    scheduler = LR_Scheduler(args.lr_scheduler, learning_rate, epochs, len(train_loader))
    optimizer_reg = optim.Adam(params_reg, lr=learning_rate, weight_decay=args.weight_decay)
    optimizer_1 = optim.Adam(params_seg_1, lr=learning_rate, weight_decay=args.weight_decay)
    optimizer_2 = optim.Adam(params_seg_2, lr=learning_rate, weight_decay=args.weight_decay)
    crfloss = DenseCRFLoss(1e-8, 20, 5, 1.0)

    print("----starting training !----------")

    best_valid_dsc = 0.0
    final_train_dsc = 0.0

    final_train_loss = 0.0
    final_valid_loss = 0.0

    for epoch in range(epochs):

        sim_loss_train = 0.0
        grad_loss_train = 0.0
        reg_loss_train = 0.0
        av_crf_loss = 0.0
        avdsc_train = 0.0
        seg_loss_train = 0.0
        weakly_loss_train = 0.0
        sup_loss_train = 0.0
        unsup_loss_train = 0.0
        total_loss_train = 0.0
        train_bar = tqdm(train_loader)
        SegNet1.train()
        SegNet2.train()
        RegModel.train()
        stn.train()

        for step, (unsup_images, sup_images, sup_labels) in enumerate(train_bar):
            _b, _c, _h, _w, _d = unsup_images.size()
            scheduler(optimizer_reg, step, epoch)
            scheduler(optimizer_1, step, epoch)
            scheduler(optimizer_2, step, epoch)

            img_fixed = unsup_images
            img_moving = sup_images
            label_moving = sup_labels.to(torch.float32)

            input_image = torch.cat([img_fixed, img_moving], dim=1)
            flow = RegModel(input_image.cuda())
            img_warped = stn(img_moving.cuda(), flow, mode='bilinear')
            label_warped = stn(label_moving.cuda(), flow, mode='nearest')

            unsup_images_list = [img_fixed[:, :, :, :, j] for j
                                 in range(_d)]
            sup_images_list = [img_moving[:, :, :, :, j] for j
                               in range(_d)]
            sup_labels_list = [label_moving[:, :, :, :, j] for j
                               in range(_d)]
            warped_images_list = [img_warped[:, :, :, :, j] for j
                                  in range(_d)]
            warped_labels_list = [label_warped[:, :, :, :, j] for j
                                  in range(_d)]
            unsup_images_for_seg = torch.cat(unsup_images_list, dim=0)
            sup_images_for_seg = torch.cat(sup_images_list, dim=0)
            sup_labels_for_seg = torch.cat(sup_labels_list, dim=0).squeeze()

            warped_images_for_seg = torch.cat(warped_images_list, dim=0)
            warped_labels_for_seg = torch.cat(warped_labels_list, dim=0).squeeze()

            aug_images, aug_labels = sup_images_for_seg.squeeze().numpy(), sup_labels_for_seg.numpy()
            aug_images, aug_labels = augment_data_batch(aug_images, aug_labels, shift=0, rotate=60, scale=0.5,
                                                        intensity=0.5,
                                                        flip=False)
            shift_max = 60
            shift = int(shift_max * np.random.uniform(-1, 1))
            aug_images = crop_batch_data(aug_images, (image_size[0], image_size[1]), shift_value=shift)
            aug_labels = crop_batch_data(aug_labels, (image_size[0], image_size[1]), shift_value=shift)

            aug_sup_images = torch.Tensor(np.expand_dims(aug_images, axis=1))
            aug_sup_labels = torch.Tensor(aug_labels).to(torch.float32)

            quotient = (_b * _d) // args.seg_batch_size
            remainder = (_b * _d) % args.seg_batch_size

            split_list = [(args.seg_batch_size if i < quotient else remainder) for i in range(quotient + 1)]
            if remainder == 0:
                split_list.pop(-1)

            for p in RegModel.parameters():
                p.requires_grad = False
            for p in SegNet1.parameters():
                p.requires_grad = True
            for p in SegNet2.parameters():
                p.requires_grad = False

            unsup_seg_images_list = torch.split(unsup_images_for_seg, split_list, dim=0)
            warped_seg_image_list = torch.split(warped_images_for_seg, split_list, dim=0)
            sup_seg_images_list = torch.split(aug_sup_images, split_list, dim=0)

            output_for_sup = torch.cat([SegNet1(images.cuda()) for images in sup_seg_images_list], dim=0)
            output_for_warped = torch.cat([SegNet1(images.cuda()) for images in warped_seg_image_list], dim=0)

            loss_1 = F.cross_entropy(output_for_sup, aug_sup_labels.long().cuda())
            loss_2 = F.cross_entropy(output_for_warped, warped_labels_for_seg.long().cuda())
            sup_loss = (loss_1 + loss_2) / 2

            sup_loss.backward(retain_graph=True)
            optimizer_1.step()
            SegNet1.zero_grad()
            optimizer_1.zero_grad()

            # warped_pseudo_labels = torch.argmax(output_for_warped, dim=1)

            for p in RegModel.parameters():
                p.requires_grad = False
            for p in SegNet1.parameters():
                p.requires_grad = False
            for p in SegNet2.parameters():
                p.requires_grad = True
            warped_labels_for_seg.detach()
            output_for_unsup = torch.cat([SegNet2(images.cuda()) for images in unsup_seg_images_list], dim=0)
            ce_loss = F.cross_entropy(output_for_unsup, warped_labels_for_seg.long().cuda())
            # crf_loss = crfloss(unsup_images_for_seg.repeat(1, 3, 1, 1), F.softmax(output_for_unsup, dim=1),
            #                    torch.ones_like(torch.argmax(unsup_images_for_seg, dim=1)).to(torch.float32)).cuda()
            crf_loss = 0.0

            loss_weakly = ce_loss + crf_loss
            loss_weakly.backward()
            optimizer_2.step()
            SegNet2.zero_grad()
            optimizer_2.zero_grad()

            unsup_pseudo_labels = torch.argmax(output_for_unsup, dim=1)

            for p in RegModel.parameters():
                p.requires_grad = True
            for p in SegNet1.parameters():
                p.requires_grad = False
            for p in SegNet2.parameters():
                p.requires_grad = False

            unsup_pseudo_labels = unsup_pseudo_labels.detach()

            unsup_loss = 0.0

            sim_loss = sim_loss_fn(img_warped, img_fixed.cuda())
            grad_loss = grad_loss_fn(flow)
            reg_loss = sim_loss + args.alpha * grad_loss + unsup_loss

            reg_loss.backward()
            optimizer_reg.step()
            RegModel.zero_grad()
            optimizer_reg.zero_grad()

            loss = reg_loss + sup_loss + loss_weakly + unsup_loss
            dsc = DSC_average(output_for_warped, warped_labels_for_seg.cuda())

            # loss.backward()
            #
            # optimizer_reg.step()
            # optimizer_1.step()
            # # optimizer_2.step()

            sim_loss_train += sim_loss.item()
            grad_loss_train += grad_loss.item()
            reg_loss_train += reg_loss.item()
            av_crf_loss += crf_loss
            weakly_loss_train += loss_weakly.item()
            sup_loss_train += sup_loss.item()
            seg_loss_train += sup_loss.item() + loss_weakly.item()
            total_loss_train += loss.item()
            avdsc_train += dsc.item()
            unsup_loss_train += unsup_loss

        RegModel.eval()
        stn.eval()
        SegNet1.eval()
        SegNet2.eval()

        avdsc_valid = 0.0
        avloss_valid = 0.0

        with torch.no_grad():
            for val_sup_images, val_sup_labels in valid_loader:
                _b, _c, _h, _w, _d = val_sup_images.size()

                val_sup_images_list = [val_sup_images[:, :, :, :, j] for j in range(_d)]
                val_sup_labels_list = [val_sup_labels[:, :, :, :, j] for j in range(_d)]

                val_images = torch.cat(val_sup_images_list, dim=0)
                val_labels = torch.cat(val_sup_labels_list, dim=0).squeeze()
                quotient = (_b * _d) // args.seg_batch_size
                remainder = (_b * _d) % args.seg_batch_size

                split_list = [(args.seg_batch_size if i < quotient else remainder) for i in range(quotient + 1)]
                if remainder == 0:
                    split_list.pop(-1)

                val_images_seg_list = torch.split(val_images, split_list, dim=0)

                outputs = torch.cat([SegNet1(images.cuda()) for images in val_images_seg_list], dim=0)
                loss = F.cross_entropy(outputs, val_labels.cuda())
                dsc = DSC_average(outputs, val_labels.cuda())

                avdsc_valid += dsc.item()
                avloss_valid += loss.item()

        avdsc_valid /= len(valid_loader)
        avloss_valid /= len(valid_loader)
        sim_loss_train /= len(train_loader)
        grad_loss_train /= len(train_loader)
        reg_loss_train /= len(train_loader)
        av_crf_loss /= len(train_loader)
        avdsc_train /= len(train_loader)
        seg_loss_train /= len(train_loader)
        weakly_loss_train /= len(train_loader)
        sup_loss_train /= len(train_loader)
        total_loss_train /= len(train_loader)
        unsup_loss_train /= len(train_loader)

        print(
            "Epoch[{}/{}] Training average total loss:{:.4f}, DSC:{:.3f}, Registration loss:{:.4f}, "
            "sup_loss:{:.3f}, weakly_loss:{:.3f}, unsup_loss:{:.4f}, Validation average loss:{:.3f}, Validation"
            "average DCS:{:.3f}".format(
                epoch + 1, epochs,
                total_loss_train, avdsc_train,
                reg_loss_train, sup_loss_train, weakly_loss_train, unsup_loss_train,
                avloss_valid, avdsc_valid
            )
        )
        if avdsc_valid > best_valid_dsc:
            best_valid_dsc = avdsc_valid
            final_train_dsc = avdsc_train
            print("model saved !")
            torch.save(RegModel.state_dict(), reg_save_path)
            torch.save(SegNet1.state_dict(), seg_save_path1)
            torch.save(SegNet2.state_dict(), seg_save_path2)

        writer.add_scalar("Train/Loss", total_loss_train, epoch)
        writer.add_scalar("Train/DSC", avdsc_train, epoch)
        writer.add_scalar("Validation/Loss", avloss_valid, epoch)
        writer.add_scalar("Validation/DSC", avdsc_valid, epoch)
    writer.close()
    print("Finish Training!")


def testing(args):
    image_size = (args.image_size[0], args.image_size[1])
    num_classes = args.num_classes

    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    pred_dir = test_output_dir + '/predictions'

    model_path = os.path.join(model_dir, 'SegNet1.pth')
    SegNet = SingleUnet('resnet50', 'imagenet', classes=num_classes, deep_stem=True)

    if torch.cuda.is_available():
        SegNet.cuda()
    SegNet.load_state_dict(torch.load(model_path, map_location=device))
    SegNet.eval()
    mkdir(pred_dir)

    seed_torch(args.seed)
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
                outputs = SegNet(tmp_image.to(device))
                tmp_prob = F.softmax(outputs, dim=1)
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_end = img_name.find('.')
            loc_start = img_name.rfind('/')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = savedirname + img_name[loc_start:loc_end] + '_Pred.nii.gz'
            nib.save(nii_pred, pred_name)

    print('Finished Testing')


def extra_test(args):
    image_size = (args.image_size[0], args.image_size[1])
    num_classes = args.num_classes

    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    pred_dir = test_output_dir + '/series_predictions'

    model_path = os.path.join(model_dir, 'SegNet1.pth')
    SegNet = SingleUnet('resnet50', 'imagenet', classes=num_classes, deep_stem=True)

    if torch.cuda.is_available():
        SegNet.cuda()
    SegNet.load_state_dict(torch.load(model_path, map_location=device))
    SegNet.eval()
    mkdir(pred_dir)

    seed_torch(args.seed)
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
                outputs = SegNet(tmp_image.to(device))
                tmp_prob = F.softmax(outputs, dim=1)
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_end = img_name.find('.')
            loc_start = img_name.rfind('/')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = savedirname + img_name[loc_start:loc_end] + '_Pred.nii.gz'
            nib.save(nii_pred, pred_name)

    print('Finished Testing')


def test_reg(args):
    from medpy import metric
    def calculate_metrics_persubj(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            assd = metric.binary.assd(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, assd, hd95
        else:
            return 0, 0, 0

    image_size = args.image_size
    num_classes = args.num_classes

    model_dir = args.train_output_dir + '/model'
    test_output_dir = args.test_output_dir
    pred_dir = test_output_dir + '/reg_predictions'
    model_path = os.path.join(model_dir, 'Registration.pth')
    RegModel = VMDiff(args.image_size, args.steps, args.downsize).cuda()
    stn = SpatialTransformer(args.image_size).cuda()

    RegModel.load_state_dict(torch.load(model_path, map_location=device))
    RegModel.eval()
    mkdir(pred_dir)

    seed_torch(args.seed)

    metrics_1 = []
    metrics_2 = []
    metrics_3 = []
    data_list = get_image_list(args.test_data_list)
    with torch.no_grad():
        for index in range(0, len(data_list['image_filenames']), 2):
            img_name = data_list['image_filenames'][index].split('/')[0].split('_')[0]
            fixed_img_name = os.path.join(args.test_data_dir, data_list['image_filenames'][index])
            fixed_gt_name = os.path.join(args.test_data_dir, data_list['label_filenames'][index])

            moving_img_name = os.path.join(args.test_data_dir, data_list['image_filenames'][index].replace('ED', 'ES'))
            moving_gt_name = os.path.join(args.test_data_dir, data_list['label_filenames'][index].replace('ED', 'ES'))

            nib_fixed_img = nib.load((os.path.join(args.test_data_dir, fixed_img_name)))
            nib_fixed_gt = nib.load((os.path.join(args.test_data_dir, fixed_gt_name)))
            nib_moving_img = nib.load((os.path.join(args.test_data_dir, moving_img_name)))
            nib_moving_gt = nib.load((os.path.join(args.test_data_dir, moving_gt_name)))

            fixed_img = nib_fixed_img.get_fdata().astype('float32')
            fixed_gt = nib_fixed_gt.get_fdata().astype('int16')

            moving_img = nib_moving_img.get_fdata().astype('float32')
            moving_gt = nib_moving_gt.get_fdata().astype('int16')

            clip_min = np.percentile(fixed_img, 1)
            clip_max = np.percentile(fixed_img, 99)
            fixed_img = np.clip(fixed_img, clip_min, clip_max)
            fixed_img = (fixed_img - fixed_img.min()) / float(fixed_img.max() - fixed_img.min())
            x, y, z = fixed_img.shape
            x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
            fixed_img = crop_3D_image(fixed_img, x_centre, y_centre, z_center, image_size, constant_values=0)

            x, y, z = fixed_gt.shape
            x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
            fixed_gt = crop_3D_image(fixed_gt, x_centre, y_centre, z_center, image_size, constant_values=0)

            clip_min = np.percentile(moving_img, 1)
            clip_max = np.percentile(moving_img, 99)
            moving_img = np.clip(moving_img, clip_min, clip_max)
            moving_img = (moving_img - moving_img.min()) / float(moving_img.max() - moving_img.min())
            x_moving, y_moving, z_moving = moving_img.shape
            x_centre, y_centre, z_center = int(x_moving / 2), int(y_moving / 2), int(z_moving / 2)
            moving_img = crop_3D_image(moving_img, x_centre, y_centre, z_center, image_size, constant_values=0)

            x, y, z = moving_gt.shape
            x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
            moving_gt = crop_3D_image(moving_gt, x_centre, y_centre, z_center, image_size, constant_values=0)

            tensor_fixed_img = torch.from_numpy(fixed_img).unsqueeze(0).unsqueeze(0).cuda()
            tensor_moving_img = torch.from_numpy(moving_img).unsqueeze(0).unsqueeze(0).cuda()
            tensor_fixed_gt = torch.from_numpy(fixed_gt).unsqueeze(0).unsqueeze(0).float().cuda()
            tensor_moving_gt = torch.from_numpy(moving_gt).unsqueeze(0).unsqueeze(0).float().cuda()

            input_image = torch.cat([tensor_fixed_img, tensor_moving_img], dim=1).cuda()
            flow = RegModel(input_image)
            img_warped = stn(tensor_moving_img, flow, mode='bilinear')
            label_warped = stn(tensor_moving_gt, flow, mode='nearest')
            metrics_1.append(calculate_metrics_persubj(label_warped.cpu().detach().numpy() == 1, tensor_fixed_gt.cpu().detach().numpy() == 1))
            metrics_2.append(calculate_metrics_persubj(label_warped.cpu().detach().numpy() == 2, tensor_fixed_gt.cpu().detach().numpy() == 2))
            metrics_3.append(calculate_metrics_persubj(label_warped.cpu().detach().numpy() == 3, tensor_fixed_gt.cpu().detach().numpy() == 3))

            warped_img = crop_3D_image(img_warped.squeeze().cpu(), image_size[0] // 2, image_size[1] // 2,
                                       image_size[2] // 2,
                                       (x_moving, y_moving, z_moving), constant_values=0)

            warped_gt = crop_3D_image(label_warped.squeeze().cpu(), image_size[0] // 2, image_size[1] // 2,
                                      image_size[2] // 2,
                                      (x_moving, y_moving, z_moving), constant_values=0)
            flow_warped = crop_3D_image(flow.squeeze().cpu(), image_size[0] // 2, image_size[1] // 2,
                                        image_size[2] // 2,
                                        (x_moving, y_moving, z_moving), constant_values=0)

            nii_warped_img = nib.Nifti1Image(warped_img, None, header=nib_moving_img.header)
            nii_warped_gt = nib.Nifti1Image(warped_gt, None, header=nib_moving_gt.header)
            nii_flow = nib.Nifti1Image(flow_warped, None, header=nib_moving_img.header)

            savedirname = os.path.join(pred_dir, img_name)
            warped_img_name = savedirname + '/dwarped.nii.gz'
            warped_gt_name = savedirname + '/gt.nii.gz'
            warped_flow_name = savedirname + '/flow.nii.gz'
            mkdir(savedirname)
            nib.save(nii_warped_img, warped_img_name)
            nib.save(nii_warped_gt, warped_gt_name)
            nib.save(nii_flow, warped_flow_name)

    metrics_1 = np.array(metrics_1)
    metrics_2 = np.array(metrics_2)
    metrics_3 = np.array(metrics_3)
    with open(os.path.join(pred_dir, 'results.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_filenames', 'dsc1', 'dsc2', 'dsc3','assd1', 'assd2', 'assd3', 'hd95_1', 'hd95_2', 'hd95_3'])
        writer.writeheader()

        for idx, subj in enumerate(range(101, 151)):
            writer.writerow({
                'image_filenames': 'patient{:03d}'.format(subj),
                'dsc1': metrics_1[idx][0],
                'dsc2': metrics_2[idx][0],
                'dsc3': metrics_3[idx][0],
                'assd1':metrics_1[idx][1],
                'assd2':metrics_2[idx][1],
                'assd3':metrics_3[idx][1],
                'hd95_1':metrics_1[idx][2],
                'hd95_2':metrics_2[idx][2],
                'hd95_3':metrics_3[idx][2],
            })
        average_metrics_1 = np.sum(metrics_1, axis=0) / len(metrics_1)
        average_metrics_2 = np.sum(metrics_2, axis=0) / len(metrics_2)
        average_metrics_3 = np.sum(metrics_3, axis=0) / len(metrics_3)
        writer.writerow({
            'image_filenames': 'average',
            'dsc1': average_metrics_1[0],
            'dsc2': average_metrics_2[0],
            'dsc3': average_metrics_3[0],
            'assd1': average_metrics_1[1],
            'assd2': average_metrics_2[1],
            'assd3': average_metrics_3[1],
            'hd95_1': average_metrics_1[2],
            'hd95_2': average_metrics_2[2],
            'hd95_3': average_metrics_3[2],
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output_dir", type=str, default=None)
    parser.add_argument("--reg_batch_size", type=int, default=2)
    parser.add_argument("--seg_batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image_size", default=(224, 224, 18))
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_sup_data_csv", type=str, default=None)
    parser.add_argument("--train_unsup_data_csv", type=str, default=None)
    parser.add_argument("--valid_sup_data_csv", type=str, default=None)
    parser.add_argument("--valid_unsup_data_csv", type=str, default=None)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--downsize", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sim_loss", type=str, default='mse')
    parser.add_argument("--lr_scheduler", type=str, default='poly')
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha",
                        default=0.01)
    parser.add_argument("--test_output_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--test_data_list", type=str, default=None)
    parser.add_argument("--mode", type=str, default='train')
    args = parser.parse_args()
    mode = args.mode

    if mode == 'train':
        training(args)
        testing(args)
    if mode == 'test':
        testing(args)
    if mode == 'external':
        extra_test(args)
    if mode == 'test_reg':
        test_reg(args)
    else:
        raise NotImplementedError

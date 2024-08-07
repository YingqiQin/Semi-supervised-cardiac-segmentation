# python imports
import argparse
import csv
import os
import time
import warnings

import nibabel as nib
import numpy as np
import skimage.metrics as metrics  # 这个里面包含了很多评估指标的计算方法 PSNR SSIM等
# external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

# 文件路径
parser.add_argument("--train_dir", type=str, help="data folder with training vols", dest="train_dir",
                    default="../../dataset/ACDC")
parser.add_argument("--model_dir", type=str, help="models folder", dest="model_dir", default="./Checkpoint_ACDC")
parser.add_argument("--log_dir", type=str, help="logs folder", dest="log_dir", default="./Log_ACDC")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default="./Result_ACDC")

# network parameters
parser.add_argument("--pattern", type=str, help="select train or test", dest="pattern", default="train")
parser.add_argument("--steps", type=int, help="number of integration steps", dest="steps", default=7)
parser.add_argument("--downsize", type=int, help="flow down sample factor for integration", dest="downsize", default=2)

# training parameters
parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations", dest="n_iter", default=1000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc", dest="sim_loss", default="mse")
parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha",
                    default=0.01)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=4)

# testing parameters
parser.add_argument("--test_dir", type=str, help="test data directory", dest="test_dir", default="../../dataset/ACDC")
parser.add_argument("--checkpoint_path", type=str, help="model weight file", dest="checkpoint_path",
                    default="./Checkpoint_ACDC/trained_model.pth")
parser.add_argument("--dice", type=bool, help="if compute dice", dest="dice", default=True)
parser.add_argument("--psnr", type=bool, help="if compute psnr", dest="psnr", default=True)
parser.add_argument("--ssim", type=bool, help="if compute ssim", dest="ssim", default=True)
parser.add_argument("--jacobian", type=bool, help="if compute jacobian", dest="jac", default=True)

args = parser.parse_args()


# mse loss
def compute_mse(tensor_x, tensor_y):
    mse = torch.mean((tensor_x - tensor_y) ** 2)
    return mse


# gradient loss
def compute_gradient(tensor_x):
    dims = tensor_x.ndim
    gradient = 0.0
    if dims == 4:
        dx = (tensor_x[:, :, 1:, :] - tensor_x[:, :, :-1, :]) ** 2
        dy = (tensor_x[:, :, :, 1:] - tensor_x[:, :, :, :-1]) ** 2
        gradient = (dx.mean() + dy.mean()) / 2
    elif dims == 5:
        dx = (tensor_x[:, :, 1:, :, :] - tensor_x[:, :, :-1, :, :]) ** 2
        dy = (tensor_x[:, :, :, 1:, :] - tensor_x[:, :, :, :-1, :]) ** 2
        dz = (tensor_x[:, :, :, :, 1:] - tensor_x[:, :, :, :, :-1]) ** 2
        gradient = (dx.mean() + dy.mean() + dz.mean()) / 3
    return gradient


def compute_local_sums(x, y, filt, stride, padding, win):
    x2, y2, xy = x * x, y * y, x * y
    x_sum = F.conv2d(x, filt, stride=stride, padding=padding)
    y_sum = F.conv2d(y, filt, stride=stride, padding=padding)
    x2_sum = F.conv2d(x2, filt, stride=stride, padding=padding)
    y2_sum = F.conv2d(y2, filt, stride=stride, padding=padding)
    xy_sum = F.conv2d(xy, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    x_windowed = x_sum / win_size
    y_windowed = y_sum / win_size
    cross = xy_sum - y_windowed * x_sum - x_windowed * y_sum + x_windowed * y_windowed * win_size
    x_var = x2_sum - 2 * x_windowed * x_sum + x_windowed * x_windowed * win_size
    y_var = y2_sum - 2 * y_windowed * y_sum + y_windowed * y_windowed * win_size
    return x_var, y_var, cross


# ncc损失
def ncc_loss(x, y, win=None):
    """
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    """
    ndims = len(list(x.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).cuda()
    pad_no = np.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    x_var, y_var, cross = compute_local_sums(x, y, sum_filt, stride, padding, win)
    cc = cross * cross / (x_var * y_var + 1e-5)
    return -1 * torch.mean(cc)


# count parameters in model
def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# 计算Dice B*C*H*W*D  多标签返回均值
def compute_Dice(tensor_pred, tensor_targ):
    smooth = 1e-5
    labels = tensor_pred.unique()
    if labels[0] == 0:
        labels = labels[1:]
    dice_list = torch.zeros([len(labels)])
    for _num in range(len(labels)):
        tensor_x = torch.where(tensor_pred == labels[_num], 1, 0).flatten()
        tensor_y = torch.where(tensor_targ == labels[_num], 1, 0).flatten()
        dice_list[_num] = (2.0 * (tensor_x * tensor_y).sum() + smooth) / (tensor_x.sum() + tensor_y.sum() + smooth)
    dice = torch.mean(dice_list).item()
    return dice


# compute the peak signal noise ratio //tensor
def compute_PSNR(tensor_x, tensor_y):
    mse = compute_mse(tensor_x, tensor_y)
    psnr = (-10 * torch.log10(mse)).item()
    return psnr


# compute structure similarity //tensor
def compute_SSIM(tensor_x, tensor_y):
    np_x = tensor_x.cpu().detach().numpy()[0, 0, ...]
    np_y = tensor_y.cpu().detach().numpy()[0, 0, ...]
    ssim = metrics.structural_similarity(np_x, np_y, data_range=1)
    return ssim


# compute Jacobian determinant
def compute_Jacobian(flow):
    Dy = (flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1])
    Dx = (flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1])
    Dz = (flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1])

    D1 = (Dx[:, 0, ...] + 1) * ((Dy[:, 1, ...] + 1) * (Dz[:, 2, ...] + 1) - Dy[:, 2, ...] * Dz[:, 1, ...])
    D2 = (Dx[:, 1, ...]) * (Dy[:, 0, ...] * (Dz[:, 2, ...] + 1) - Dy[:, 2, ...] * Dz[:, 0, ...])
    D3 = (Dx[:, 2, ...]) * (Dy[:, 0, ...] * Dz[:, 1, ...] - (Dy[:, 1, ...] + 1) * Dz[:, 0, ...])

    D = D1 - D2 + D3
    return D


class Jacobian:
    def __init__(self, flow):
        self.determinant = compute_Jacobian(flow)

    def count_minus_ratio(self):
        size = 1
        for dim in self.determinant.shape:
            size *= dim
        x = torch.where(self.determinant <= 0, 1, 0)
        ratio = (torch.sum(x) / size).item()
        return ratio


# Unet模块
class UnetBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, img_in):
        img_out = self.conv1(img_in)
        img_out = self.bn1(img_out)
        img_out = self.relu(img_out)

        img_out = self.conv2(img_out)
        img_out = self.bn2(img_out)
        img_out = self.relu(img_out)

        return img_out


# Unet网络
class UNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = (16, 32, 32, 32, 32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up4 = nn.Upsample(size=(256, 256, 18), mode='trilinear', align_corners=True)
        self.up3 = nn.Upsample(size=(128, 128, 9), mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(size=(64, 64, 4), mode='trilinear', align_corners=True)
        self.up1 = nn.Upsample(size=(32, 32, 2), mode='trilinear', align_corners=True)

        self.enc1 = UnetBlock(input_channels, nb_filter[0], nb_filter[0])
        self.enc2 = UnetBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.enc3 = UnetBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.enc4 = UnetBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.enc5 = UnetBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.dec1 = UnetBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.dec2 = UnetBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.dec3 = UnetBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.dec4 = UnetBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.dec5 = nn.Conv3d(nb_filter[0], output_channels, kernel_size=1)

    def forward(self, img_in):
        # unet
        img_enc1 = self.enc1(img_in)
        img_enc2 = self.enc2(self.pool(img_enc1))
        img_enc3 = self.enc3(self.pool(img_enc2))
        img_enc4 = self.enc4(self.pool(img_enc3))
        img_enc5 = self.enc5(self.pool(img_enc4))

        img_dec1 = self.dec1(torch.cat([img_enc4, self.up1(img_enc5)], dim=1))
        img_dec2 = self.dec2(torch.cat([img_enc3, self.up2(img_dec1)], dim=1))
        img_dec3 = self.dec3(torch.cat([img_enc2, self.up3(img_dec2)], dim=1))
        img_dec4 = self.dec4(torch.cat([img_enc1, self.up4(img_dec3)], dim=1))
        # img_out = self.dec5(img_dec4)
        img_out = img_dec4

        return img_out


# STN空间变换网络
class SpatialTransformer(nn.Module):
    def __init__(self, img_shape):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid  B*C*H*W*D
        vectors = [torch.arange(0, s) for s in img_shape]
        grid = torch.stack(torch.meshgrid(vectors)).unsqueeze(0).type(torch.float32)
        self.register_buffer('grid', grid)

    def forward(self, img_moving, flow, mode='bilinear'):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        img_warped = F.grid_sample(img_moving, new_locs, align_corners=True, mode=mode)
        return img_warped


# 重采样
class Resize(nn.Module):
    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")

        return x


# 积分
class Integrate(nn.Module):
    def __init__(self, in_shape, steps):
        super().__init__()
        self.steps = steps
        self.scale = 1.0 / (2 ** self.steps)
        self.transformer = SpatialTransformer(in_shape)

    def forward(self, vector):
        vector = vector * self.scale
        for _ in range(self.steps):
            vector = vector + self.transformer(vector, vector)
        return vector


# 微分同胚VM模型
class VMDiff(nn.Module):
    def __init__(self, in_shape, int_steps=7, int_downsize=2):
        super().__init__()
        # 包含unet 去掉了最后一层卷积输出 现在的channels=16
        self.unet = UNet(input_channels=2)
        # 一个卷积层 对feature进行采样
        self.flow = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # resize
        self.resize = Resize(int_downsize)
        # 恢复到原来的大小
        self.full_size = Resize(1.0 / int_downsize)
        # 积分层
        down_shape = [int(dim / int_downsize) for dim in in_shape]
        self.integrate = Integrate(down_shape, int_steps)

    def forward(self, in_img):
        # 经过unet
        feature = self.unet(in_img)
        # 经过flow 卷积层
        flow_field = self.flow(feature)
        # resize
        pos_flow = self.resize(flow_field)
        # 积分得到形变场
        pos_flow = self.integrate(pos_flow)
        pos_flow = self.full_size(pos_flow)

        return pos_flow


def train():
    # 准备工作
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 创建log文件 命名：迭代次数_学习率_正则化系数
    print("----Make log file----")
    log_name = "%d_%lf_%f.csv" % (args.n_iter, args.lr, args.alpha)
    print("log_name: ", log_name)
    file_log = open(os.path.join(args.log_dir, log_name), "w")
    print("iter,train_loss,sim_loss,grad_loss,valid_loss,sim_loss,grad_loss,valid_dice", file=file_log)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    img_shape = [256, 256, 18]
    model = VMDiff(img_shape, args.steps, args.downsize).cuda()
    stn = SpatialTransformer(img_shape).cuda()  # 创建stn需要shape
    # 模型参数个数
    print("VMDiff: ", countParameters(model))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    opt = Adam(model.parameters(), lr=args.lr)
    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    else:
        sim_loss_fn = ncc_loss
    grad_loss_fn = compute_gradient

    # 数据处理
    print("----Process data----")

    # 数据集划分 train50 valid20
    train_list = np.arange(0, 2)
    valid_list = np.arange(50, 70)

    dataset_train_img = torch.zeros([2, 50, 256, 256, 18], dtype=torch.float32)
    dataset_valid_img = torch.zeros([2, 20, 256, 256, 18], dtype=torch.float32)
    dataset_valid_label = torch.zeros([2, 20, 256, 256, 18], dtype=torch.int8)

    subject_forms = ["ED", "ES"]
    # ED或ES
    for _form in range(2):
        # 训练集
        for _num in range(len(train_list)):
            subject = train_list[_num] + 1
            file_dir = os.path.join(args.train_dir, "patient%03d" % subject)
            # img
            file_path = os.path.join(file_dir, "patient{:03d}_{}.nii.gz".format(subject, subject_forms[_form]))
            data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
            dataset_train_img[_form, _num, :, :, :] = (data - data.min()) / (data.max() - data.min())
        # 验证集
        for _num in range(len(valid_list)):
            subject = valid_list[_num] + 1
            file_dir = os.path.join(args.train_dir, "P%03d-%s" % (subject, subject_forms[_form]))
            # img
            file_path = os.path.join(file_dir, "img_llw.nii.gz")
            data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
            dataset_valid_img[_form, _num, :, :, :] = (data - data.min()) / (data.max() - data.min())
            # label
            file_path = os.path.join(file_dir, "label_llw.nii.gz")
            data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.int8)
            dataset_valid_label[_form, _num, :, :, :] = data[:, :, :]

    # 开始训练
    print("----Start training----")
    # 计时
    start_time = float(time.time())

    best_valid_dice = 0.0
    final_train_loss = 0.0
    final_valid_loss = 0.0
    for _iter in range(1, args.n_iter + 1):
        # 将train_data_list进行随机排序
        train_list_permuted = np.random.permutation(train_list)

        # 训练部分
        sim_loss_train = 0.0
        grad_loss_train = 0.0
        loss_train = 0.0
        model.train()
        stn.train()
        # 以batch_size为步长批量读取数据
        steps = len(train_list_permuted) // args.batch_size
        for _step in range(steps):
            # 预先定义fixed 和 moving 张量 batch_size*C*H*W*D
            img_fixed = torch.zeros([args.batch_size, 1, 256, 256, 18], dtype=torch.float32)
            img_moving = torch.zeros([args.batch_size, 1, 256, 256, 18], dtype=torch.float32)

            # 迭代读取fixed 和 moving图像
            for _batch in range(args.batch_size):
                subject = _step * args.batch_size + _batch
                img_fixed[_batch, 0, :, :, :] = dataset_train_img[0, subject, :, :, :]
                img_moving[_batch, 0, :, :, :] = dataset_train_img[1, subject, :, :, :]

            img_fixed = img_fixed.cuda()
            img_moving = img_moving.cuda()

            # 先做拼接再输入网络
            input_image = torch.cat([img_fixed, img_moving], dim=1)
            flow = model(input_image)
            img_warped = stn(img_moving, flow, mode='bilinear')

            # 计算loss
            sim_loss = sim_loss_fn(img_warped, img_fixed)
            grad_loss = grad_loss_fn(flow)
            loss = sim_loss + args.alpha * grad_loss

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            sim_loss_train += sim_loss.item()
            grad_loss_train += grad_loss.item()
            loss_train += loss.item()

        sim_loss_train /= steps
        grad_loss_train /= steps
        loss_train /= steps

        # 验证部分
        sim_loss_valid = 0.0
        grad_loss_valid = 0.0
        loss_valid = 0.0
        dice_valid = 0.0
        model.eval()
        stn.eval()
        with torch.no_grad():
            img_fixed = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
            img_moving = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
            label_fixed = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
            label_moving = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)

            for _num in range(len(valid_list)):
                # img & label
                img_fixed[0, 0, :, :, :] = dataset_valid_img[0, _num, :, :, :]
                img_moving[0, 0, :, :, :] = dataset_valid_img[1, _num, :, :, :]
                label_fixed[0, 0, :, :, :] = dataset_valid_label[0, _num, :, :, :]
                label_moving[0, 0, :, :, :] = dataset_valid_label[1, _num, :, :, :]
                img_fixed = img_fixed.cuda()
                img_moving = img_moving.cuda()

                # 做拼接后输入网络
                input_image = torch.cat([img_fixed, img_moving], dim=1)
                flow = model(input_image)
                img_warped = stn(img_moving, flow, mode='bilinear')

                # 计算loss
                sim_loss = sim_loss_fn(img_warped, img_fixed)
                grad_loss = grad_loss_fn(flow)
                loss = sim_loss + args.alpha * grad_loss

                sim_loss_valid += sim_loss.item()
                grad_loss_valid += grad_loss.item()
                loss_valid += loss.item()

                # 计算dice
                label_fixed = label_fixed.cuda()
                label_moving = label_moving.cuda()
                label_warped = stn(label_moving, flow, mode='nearest')
                dice_valid += compute_Dice(label_warped, label_fixed).mean().item()

        sim_loss_valid /= len(valid_list)
        grad_loss_valid /= len(valid_list)
        loss_valid /= len(valid_list)
        dice_valid /= len(valid_list)
        print("epoch: %d  train_loss: %f  sim_loss: %f  grad_loss: %f" % (
        _iter, loss_train, sim_loss_train, grad_loss_train), flush=True)
        print("epoch: %d  valid_loss: %f  sim_loss: %f  grad_loss: %f  valid_dice: %f" % (
        _iter, loss_valid, sim_loss_valid, grad_loss_valid, dice_valid), flush=True)
        print("%d,%f,%f,%f,%f,%f,%f,%f" % (
        _iter, loss_train, sim_loss_train, grad_loss_train, loss_valid, sim_loss_valid, grad_loss_valid, dice_valid),
              file=file_log)

        # 计时
        if _iter % 10 == 0:
            print("----time_used: %f" % float(time.time() - start_time), flush=True)
            print("----time_used: %f" % float(time.time() - start_time), file=file_log)

        # 保存最佳模型参数
        if dice_valid > best_valid_dice:
            best_valid_dice = dice_valid
            final_train_loss = loss_train
            final_valid_loss = loss_valid
            # Save model checkpoint
            save_file_dir = os.path.join(args.model_dir, "trained_model.pth")
            torch.save(model.state_dict(), save_file_dir)

    print("final_train_loss = %f,final_valid_loss = %f,best_valid_dice = %f" % (
    final_train_loss, final_valid_loss, best_valid_dice), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f,best_valid_dice = %f" % (
    final_train_loss, final_valid_loss, best_valid_dice), file=file_log)
    file_log.close()


def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    img_shape = [256, 256, 18]
    model = VMDiff(img_shape, args.steps, args.downsize).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    stn = SpatialTransformer(img_shape).cuda()
    model.eval()
    stn.eval()

    # 数据处理
    print("----Process data----")
    # 测试序列
    test_list = np.arange(70, 100)

    # 读取图像数据
    dataset_img = torch.zeros([2, 30, 256, 256, 18], dtype=torch.float32)
    dataset_label = torch.zeros([2, 30, 256, 256, 18], dtype=torch.int8)
    subject_forms = ["ED", "ES"]

    for _form in range(2):
        # 测试集
        for _num in range(len(test_list)):
            subject = test_list[_num] + 1
            file_dir = os.path.join(args.test_dir, "P%03d-%s" % (subject, subject_forms[_form]))
            # img
            file_path = os.path.join(file_dir, "img_llw.nii.gz")
            data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
            dataset_img[_form, _num, :, :, :] = (data - data.min()) / (data.max() - data.min())
            # label
            file_path = os.path.join(file_dir, "label_llw.nii.gz")
            data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.int8)
            dataset_label[_form, _num, :, :, :] = data[:, :, :]

    # 开始测试
    print("----Start testing----")
    # 计时
    time_list = []
    dice_list = []
    psnr_list = []
    ssim_list = []
    jac_list = []

    img_fixed = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
    img_moving = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
    label_fixed = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)
    label_moving = torch.zeros([1, 1, 256, 256, 18], dtype=torch.float32)

    for _num in range(len(test_list)):
        # 创建subject文件目录
        subject = test_list[_num] + 1
        subject_dir = os.path.join(args.result_dir, "P%03d" % subject)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        # img & label
        img_fixed[0, 0, :, :, :] = dataset_img[0, _num, :, :, :]
        img_moving[0, 0, :, :, :] = dataset_img[1, _num, :, :, :]
        label_fixed[0, 0, :, :, :] = dataset_label[0, _num, :, :, :]
        label_moving[0, 0, :, :, :] = dataset_label[1, _num, :, :, :]
        img_fixed = img_fixed.cuda()
        img_moving = img_moving.cuda()
        label_fixed = label_fixed.cuda()
        label_moving = label_moving.cuda()

        # 做拼接后输入网络 计时
        input_image = torch.cat([img_fixed, img_moving], dim=1)
        start_time = time.time()

        flow = model(input_image)
        img_warped = stn(img_moving, flow)

        time_list.append([float(time.time() - start_time)])
        label_warped = stn(label_moving, flow, mode='nearest')

        # 计算dice
        if args.dice:
            dice_list.append([compute_Dice(label_fixed, label_moving), compute_Dice(label_fixed, label_warped)])
        # 计算psnr
        if args.psnr:
            psnr_list.append([compute_PSNR(img_fixed, img_moving), compute_PSNR(img_fixed, img_warped)])
        # 计算ssim
        if args.ssim:
            ssim_list.append([compute_SSIM(img_fixed, img_moving), compute_SSIM(img_fixed, img_warped)])
        # 计算雅克比行列式分数
        if args.jac:
            jac = Jacobian(flow)
            jac_list.append([jac.count_minus_ratio()])

        # 保存图像
        # img & label
        img = nib.Nifti1Image(img_fixed[0, 0, :, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "fixed.nii.gz"))
        label = nib.Nifti1Image(label_fixed[0, 0, :, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_fixed.nii.gz"))

        img = nib.Nifti1Image(img_moving[0, 0, :, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "moving.nii.gz"))
        label = nib.Nifti1Image(label_moving[0, 0, :, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_moving.nii.gz"))

        img = nib.Nifti1Image(img_warped[0, 0, :, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "warped.nii.gz"))
        label = nib.Nifti1Image(label_warped[0, 0, :, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_warped.nii.gz"))

        DVF = nib.Nifti1Image(flow[0, :, :, :, :].cpu().detach().numpy(), None)
        nib.save(DVF, os.path.join(subject_dir, "flow.nii.gz"))

    print("time_used = %f" % np.sum(time_list))

    # 保存结果
    with open(os.path.join(args.result_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        header = ["time"]
        data = np.array(time_list)
        if args.dice:
            header.append("dice_pre")
            header.append("dice_done")
            dice_list = np.array(dice_list)
            data = np.append(data, dice_list, axis=1)
        if args.psnr:
            header.append("psnr_pre")
            header.append("psnr_done")
            psnr_list = np.array(psnr_list)
            data = np.append(data, psnr_list, axis=1)
        if args.ssim:
            header.append("ssim_pre")
            header.append("ssim_done")
            ssim_list = np.array(ssim_list)
            data = np.append(data, ssim_list, axis=1)
        if args.jac:
            header.append("jac")
            jac_list = np.array(jac_list)
            data = np.append(data, jac_list, axis=1)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == "__main__":
    if args.pattern == "train":
        train()
    else:
        test()
    print("end")

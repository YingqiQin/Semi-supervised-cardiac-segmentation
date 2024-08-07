# python imports
import time
import csv
import os
import warnings
import argparse
# external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
import torch.utils.data as Data
import numpy as np
import SimpleITK as sitk

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default="./Result")
parser.add_argument("--pattern", type=str, help="select train or test", dest="pattern", default="train")

# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols", dest="train_dir",
                    default="../../dataset/ACDC")
parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations", dest="n_iter", default=1000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc", dest="sim_loss", default="mse")
parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha",
                    default=0.005)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=20)
parser.add_argument("--model_dir", type=str, help="models folder", dest="model_dir", default="./Checkpoint")
parser.add_argument("--log_dir", type=str, help="logs folder", dest="log_dir", default="./Log")

# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory", dest="test_dir", default="../../dataset/ACDC")
parser.add_argument("--checkpoint_path", type=str, help="model weight file", dest="checkpoint_path",
                    default="./Checkpoint/trained_model.pth")

args = parser.parse_args()


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


# 梯度损失
def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    # if (penalty == 'l2'):
    #     dy = dy * dy
    #     dx = dx * dx
    #     dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


# mse损失
def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


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


# Unet模块
class UnetBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


# Unet网络
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [16, 32, 32, 32, 32]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = UnetBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = UnetBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = UnetBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = UnetBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = UnetBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = UnetBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = UnetBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = UnetBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = UnetBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def forward(self, _input):
        x0_0 = self.conv0_0(_input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# STN空间变换网络
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
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

        return F.grid_sample(src, new_locs, mode=self.mode)


def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def saveImage(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


# 读取数据
def loadData(data_dir, file_list, order):
    file_dir = os.path.join(data_dir, file_list[order, 1], "img", file_list[order, 3])
    img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(file_dir)))
    img = (img - img.min()) / (img.max() - img.min())  # 做归一化到0-1
    img = img.unsqueeze(0).unsqueeze(0)
    return img


def train():
    # 准备工作
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建log文件 命名：迭代次数_学习率_正则化系数
    print("----Make file_log file----")
    log_name = "%d_%lf_%f.csv" % (args.n_iter, args.lr, args.alpha)
    print("log_name: ", log_name)
    file_log = open(os.path.join(args.log_dir, log_name), "w")
    print("iter, train_loss, sim_loss, grad_loss, valid_loss, sim_loss, grad_loss", file=file_log)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    unet = UNet(input_channels=2, output_channels=2).cuda()
    stn = SpatialTransformer([256, 256]).cuda()  # 创建stn需要shape
    # 模型参数个数
    print("unet: ", countParameters(unet))
    print("stn: ", countParameters(stn))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    opt = Adam(unet.parameters(), lr=args.lr)
    if args.sim_loss == "mse":
        sim_loss_fn = mse_loss
    else:
        sim_loss_fn = ncc_loss
    grad_loss_fn = gradient_loss

    # 数据处理
    print("----Process data----")
    # # 读取文件目录file_list.csv
    # file_list = []
    # with open(os.path.join(args.train_dir, "file_list.csv"), mode='r', encoding="utf-8-sig") as _file:
    #     reader = csv.reader(_file)
    #     for row in reader:
    #         file_list.append(row)
    # file_list = np.array(file_list)

    # 数据集划分
    train_list = list(range(0, 50))
    valid_list = list(range(50, 69))
    # test_list = list(range(70, 99))
    # 训练序列
    train_data_list = np.arange(0, len(train_list) * 10)
    # train_data_list = []
    # for _subject in train_list:
    #     fixed_order = 30 * _subject + 1
    #     for _slice in range(1, 30):
    #         moving_order = fixed_order+_slice
    #         train_data_list.append([fixed_order, moving_order])
    # train_data_list = np.array(train_data_list)

    # 读取图像数据 dataset = [f/m, subject, z, x, y]
    dataset_img = torch.zeros([2, 100, 10, 256, 256], dtype=torch.float32)
    dataset_label = torch.zeros([2, 100, 10, 256, 256], dtype=torch.int8)
    file_dict = ["ED", "ES"]

    for _form in range(2):

        file_dir = os.path.join(args.train_dir, file_dict[_form], "img")
        for _subject in range(100):
            data_dir = os.path.join(file_dir, "img%03d.nii.gz" % _subject)
            img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(data_dir)))
            dataset_img[_form, _subject, :, :, :] = (img - img.min()) / (img.max() - img.min())

        file_dir = os.path.join(args.train_dir, file_dict[_form], "label")
        for _subject in range(100):
            data_dir = os.path.join(file_dir, "GT%03d.nii.gz" % _subject)
            label = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(data_dir)))
            dataset_label[_form, _subject, :, :, :] = label

    # 开始训练
    print("----Start training----")
    # 计时
    start_time = float(time.time())

    best_valid_loss = 10.0
    final_train_loss = 10.0
    for _iter in range(1, args.n_iter + 1):
        # 将train_data_list进行随机排序
        data_list = np.random.permutation(train_data_list)

        # 训练部分
        sim_loss_train = 0.0
        grad_loss_train = 0.0
        loss_train = 0.0
        unet.train()
        # 以batch_size为步长批量读取数据
        steps = len(data_list) // args.batch_size
        for _step in range(steps):
            # 预先定义fixed 和 moving 张量 batch_size*C*H*W
            img_fixed = torch.zeros([args.batch_size, 1, 256, 256])
            img_moving = torch.zeros([args.batch_size, 1, 256, 256])

            # 迭代读取fixed 和 moving图像
            for _num in range(args.batch_size):
                order = _step * args.batch_size + _num
                num_subject = data_list[order] // 10
                num_slice = data_list[order] % 10
                img_fixed[_num, 0, :, :] = dataset_img[0, num_subject, num_slice, :, :]
                img_moving[_num, 0, :, :] = dataset_img[1, num_subject, num_slice, :, :]

            img_fixed = img_fixed.cuda()
            img_moving = img_moving.cuda()

            # 先做拼接再输入网络
            input_image = torch.cat([img_fixed, img_moving], dim=1)
            flow = unet(input_image)
            img_warped = stn(img_moving, flow)

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
        # print("epoch: %d  loss: %f  sim: %f  grad: %f" % (_iter, loss_train, sim_loss_train, grad_loss_train), flush=True)
        # print("%d, %f, %f, %f" % (_iter, loss_train, sim_loss_train, grad_loss_train), file=file_log)

        # 验证部分
        sim_loss_valid = 0.0
        grad_loss_valid = 0.0
        loss_valid = 0.0
        unet.eval()
        with torch.no_grad():
            for _subject in valid_list:
                img_fixed = torch.zeros([10, 1, 256, 256])
                img_moving = torch.zeros([10, 1, 256, 256])

                # 以每个subject的第一个slice作为fixed图像
                img_fixed[:, 0, :, :] = dataset_img[0, _subject, :, :, :]
                img_moving[:, 0, :, :] = dataset_img[1, _subject, :, :, :]
                img_fixed = img_fixed.cuda()
                img_moving = img_moving.cuda()

                # 做拼接后输入网络
                input_image = torch.cat([img_fixed, img_moving], dim=1)
                flow = unet(input_image)
                img_warped = stn(img_moving, flow)

                # 计算loss
                sim_loss = sim_loss_fn(img_warped, img_fixed)
                grad_loss = grad_loss_fn(flow)
                loss = sim_loss + args.alpha * grad_loss

                sim_loss_valid += sim_loss.item()
                grad_loss_valid += grad_loss.item()
                loss_valid += loss.item()

        sim_loss_valid /= len(valid_list)
        grad_loss_valid /= len(valid_list)
        loss_valid /= len(valid_list)
        print("epoch: %d  train_loss: %f  sim_loss: %f  grad_loss: %f" % (
        _iter, loss_train, sim_loss_train, grad_loss_train), flush=True)
        print("epoch: %d  valid_loss: %f  sim_loss: %f  grad_loss: %f" % (
        _iter, loss_valid, sim_loss_valid, grad_loss_valid), flush=True)
        print("%d, %f, %f, %f, %f, %f, %f" % (
        _iter, loss_train, sim_loss_train, grad_loss_train, loss_valid, sim_loss_valid, grad_loss_valid), file=file_log)

        # 计时
        if _iter % 10 == 0:
            print("----time_used: %f" % float(time.time() - start_time), flush=True)
            print("----time_used: %f" % float(time.time() - start_time), file=file_log)

        # 保存最佳模型参数
        if loss_valid < best_valid_loss:
            best_valid_loss = loss_valid
            final_train_loss = loss_train
            # Save model checkpoint
            save_file_dir = os.path.join(args.model_dir, "trained_model.pth")
            torch.save(unet.state_dict(), save_file_dir)

    print("final_train_loss = %f, best_valid_loss = %f" % (final_train_loss, best_valid_loss), flush=True)
    print("final_train_loss = %f, best_valid_loss = %f" % (final_train_loss, best_valid_loss), file=file_log)
    file_log.close()


def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    unet = UNet(input_channels=2, output_channels=2).cuda()
    unet.load_state_dict(torch.load(args.checkpoint_path))
    stn_img = SpatialTransformer([256, 256]).cuda()
    unet.eval()
    stn_img.eval()

    # 数据处理
    print("----Process data----")
    # 读取文件目录file_list.csv
    # file_list = []
    # with open(os.path.join(args.train_dir, "file_list.csv"), mode='r', encoding="utf-8-sig") as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         file_list.append(row)
    # file_list = np.array(file_list)

    # 测试序列
    test_list = list(range(70, 100))

    # 读取图像数据
    # spacing = []
    dataset_img = torch.zeros([2, 100, 10, 256, 256], dtype=torch.float32)
    dataset_label = torch.zeros([2, 100, 10, 256, 256], dtype=torch.int8)
    file_dict = ["ED", "ES"]

    for _form in range(2):

        file_dir = os.path.join(args.train_dir, file_dict[_form], "img")
        for _subject in range(100):
            data_dir = os.path.join(file_dir, "img%03d.nii.gz" % _subject)
            img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(data_dir)))
            dataset_img[_form, _subject, :, :, :] = (img - img.min()) / (img.max() - img.min())

        file_dir = os.path.join(args.train_dir, file_dict[_form], "label")
        for _subject in range(100):
            data_dir = os.path.join(file_dir, "GT%03d.nii.gz" % _subject)
            label = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(data_dir)))
            dataset_label[_form, _subject, :, :, :] = label

    # 开始测试
    print("----Start testing----")
    # 计时
    time_list = np.zeros([30])
    n = 0
    for _subject in test_list:
        # 创建subject文件目录
        subject_dir = os.path.join(args.result_dir, "P%03d" % _subject)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        # 预先定义待配准图像img和形变图像warped
        # img_array = torch.zeros([10, 256, 256])
        # warped_array = torch.zeros([10, 256, 256])
        # img_array[:, :, :] = dataset_img[1, _subject, :, :, :]

        img_fixed = torch.zeros([10, 1, 256, 256])
        img_moving = torch.zeros([10, 1, 256, 256])

        # 以每个subject的第一个slice作为fixed图像
        img_fixed[:, 0, :, :] = dataset_img[0, _subject, :, :, :]
        img_moving[:, 0, :, :] = dataset_img[1, _subject, :, :, :]
        img_fixed = img_fixed.cuda()
        img_moving = img_moving.cuda()

        # 做拼接后输入网络 计时
        input_image = torch.cat([img_fixed, img_moving], dim=1)
        start_time = time.time()

        flow = unet(input_image)
        img_warped = stn_img(img_moving, flow)

        time_list[n] = time.time() - start_time
        n = n + 1
        # warped_array[1:30, :, :] = img_warped[:, 0, :, :].cpu().detach()

        # 保存图像
        # 图像
        img_dir = os.path.join(subject_dir, "fixed.nii.gz")
        img = sitk.GetImageFromArray(img_fixed[:, 0, :, :].cpu().detach().numpy())
        # img.SetSpacing(spacing)
        sitk.WriteImage(img, img_dir)

        img_dir = os.path.join(subject_dir, "moving.nii.gz")
        img = sitk.GetImageFromArray(img_moving[:, 0, :, :].cpu().detach().numpy())
        # img.SetSpacing(spacing)
        sitk.WriteImage(img, img_dir)

        img_dir = os.path.join(subject_dir, "warped.nii.gz")
        img = sitk.GetImageFromArray(img_warped[:, 0, :, :].cpu().detach().numpy())
        # warped.SetSpacing(spacing)
        sitk.WriteImage(img, img_dir)

    print("time_used = %f" % np.sum(time_list))
    with open(os.path.join(args.result_dir, "result.txt"), "w") as f:
        print(time_list, file=f)


if __name__ == "__main__":
    if args.pattern == "train":
        train()
    else:
        test()
    print("end")

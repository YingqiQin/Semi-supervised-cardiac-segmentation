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
from image_utils import LR_Scheduler, crop_image
import torch.nn.functional as F
from TemporalPerceiver import *
from Lossmetrics import *
from Load_Data import get_image_list, augment_data_batch, crop_batch_data
import nibabel as nib
import skimage.metrics as metrics

device = torch.device('cuda:0')


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


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


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
    epochs = args.epochs
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
                                   image_size=args.image_size, mode='train')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print("Dataset Initialized!")

    writer = SummaryWriter(training_graph)

    seg_save_path = os.path.join(model_dir, 'TPer.pth')

    print("----Initial TPer Network----")

    TPerceiver = TAtten(1, 3).cuda()
    stn = SpatialTransformer([224, 224, 18]).cuda()
    sim_loss_fn = compute_mse
    grad_loss_fn = compute_gradient

    param = [p for p in TPerceiver.parameters() if p.requires_grad]
    optimizer = optim.Adam(param, lr=learning_rate, weight_decay=args.weight_decay)
    print('------starting training !-----------')

    best_valid_loss = 1000000
    final_train_dsc = 0.0

    final_train_loss = 0.0
    final_valid_loss = 0.0

    for epoch in range(epochs):
        avdsc_train = 0.0
        avloss_train = 0.0
        sim_loss_train = 0.0
        grad_loss_train = 0.0
        loss_train = 0.0

        train_bar = tqdm(train_loader)
        TPerceiver.train()
        stn.train()
        for step, (unsup_images, sup_images, sup_labels) in enumerate(train_bar):
            img_fixed = unsup_images.cuda()
            img_moving = sup_images.cuda()

            optimizer.zero_grad()
            flow = TPerceiver([img_fixed, img_moving])
            img_warped = stn(img_moving, flow, mode='bilinear')

            sim_loss = sim_loss_fn(img_warped, img_fixed)
            grad_loss = grad_loss_fn(flow)
            loss = sim_loss + 0.01 * grad_loss

            loss.backward()
            optimizer.step()

            sim_loss_train += sim_loss.item()
            grad_loss_train += grad_loss.item()

            avloss_train += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        sim_loss_train /= len(train_loader)
        grad_loss_train /= len(train_loader)
        avloss_train /= len(train_loader)

        sim_loss_valid = 0.0
        grad_loss_valid = 0.0
        loss_valid = 0.0

        avdsc_valid = 0.0
        avloss_valid = 0.0
        TPerceiver.eval()
        stn.eval()
        with torch.no_grad():
            for val_unspu_images, val_sup_images, val_sup_labels in valid_loader:
                img_fixed = val_unspu_images.cuda()
                img_moving = val_sup_images.cuda()
                val_sup_labels = val_sup_labels.squeeze()
                flow = TPerceiver([img_fixed, img_moving])
                img_warped = stn(img_moving, flow, mode='bilinear')

                sim_loss = sim_loss_fn(img_warped, img_fixed)
                grad_loss = grad_loss_fn(flow)
                loss = sim_loss + 0.01 * grad_loss

                avloss_valid += loss.item()
        avloss_valid /= len(valid_loader)
        print(
            "train epoch[{}/{}] Training average loss:{:.4f}, Validation average loss:{:.4f}".format(
                epoch + 1, epochs,
                avloss_train,
                avloss_valid))
        if avloss_valid < best_valid_loss:
            best_valid_loss = avloss_valid
            final_train_dsc = avdsc_train
            print("model saved!")
            torch.save(TPerceiver.state_dict(), seg_save_path)
        writer.add_scalar("UNet++ Train/Loss", avloss_train, epoch)
        writer.add_scalar("UNet++ Train/DSC", avdsc_train, epoch)
        writer.add_scalar("UNet++ Validation/Loss", avloss_valid, epoch)
        writer.add_scalar("UNet++ Validation/DSC", avdsc_valid, epoch)

    print()
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image_size", default=(224, 224, 18))
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_sup_data_csv", type=str, default=None)
    parser.add_argument("--train_unsup_data_csv", type=str, default=None)
    parser.add_argument("--valid_sup_data_csv", type=str, default=None)
    parser.add_argument("--valid_unsup_data_csv", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()

    training(args)

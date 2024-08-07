import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TPer', 'TAtten']

class TemporalAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalAtten, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1_1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv_1_2 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.conv_2 = nn.Conv3d(out_channels, out_channels, 1)

        self.conv_3_1 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.conv_3_2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

    def forward(self, x_list):
        x1, x2 = x_list
        x_sum = self.conv_1_1(x1) + self.conv_1_2(x2)
        x_sum = F.adaptive_avg_pool3d(x_sum, 1)
        x_avg = F.relu(self.conv_2(x_sum))
        x1_weight = x1 * x_avg
        x2_weight = x2 * x_avg

        x1_weight = F.relu(self.conv_3_1(x1_weight))
        x2_weight = F.relu(self.conv_3_2(x2_weight))

        x = x1_weight + x2_weight

        return x
class ConvGRU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvGRU, self).__init__()

        self.conv_xz = nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv_xr = nn.Conv3d(in_channel, out_channel, 3, padding=1)
        self.conv_xn = nn.Conv3d(in_channel, out_channel, 3, padding=1)

        self.conv_hz = nn.Conv3d(out_channel, out_channel, 3, padding=1)
        self.conv_hr = nn.Conv3d(out_channel, out_channel, 3, padding=1)
        self.conv_hn = nn.Conv3d(out_channel, out_channel, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(h)
        return h, h


class ConvBlock(nn.Module):
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


class TPer(nn.Module):
    def __init__(self, in_channels, out_channels, mode='Pretrained'):
        super(TPer, self).__init__()

        self.mode = mode

        nb_filter = (16, 32, 32, 32, 32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up4 = nn.Upsample(size=(224, 224, 18), mode='trilinear', align_corners=True)
        self.up3 = nn.Upsample(size=(112, 112, 9), mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(size=(56, 56, 4), mode='trilinear', align_corners=True)
        self.up1 = nn.Upsample(size=(28, 28, 2), mode='trilinear', align_corners=True)

        self.enc1 = ConvBlock(in_channels, nb_filter[0], nb_filter[0])
        self.enc2 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.enc3 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.enc4 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.enc5 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.gru = ConvGRU(nb_filter[4], nb_filter[4])

        self.dec1 = ConvBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.dec2 = ConvBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.dec3 = ConvBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.dec4 = ConvBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.dec5 = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x_list):
        x_1, x_2 = x_list

        x1 = self.enc1(x_1)
        x1 = self.enc2(self.pool(x1))
        x1 = self.enc3(self.pool(x1))
        x1 = self.enc4(self.pool(x1))
        x1 = self.enc5(self.pool(x1))

        x1, h1 = self.gru(x1)

        x2 = self.enc1(x_2)
        x2 = self.enc2(self.pool(x2))
        x2 = self.enc3(self.pool(x2))
        x2 = self.enc4(self.pool(x2))
        x2 = self.enc5(self.pool(x2))

        x2, _ = self.gru(x2, h1)

        if self.mode == 'Pretrained':
            img = self.dec1(self.up1(x2))
            img = self.dec2(self.up2(img))
            img = self.dec3(self.up3(img))
            img = self.dec4(self.up4(img))
            img = self.dec5(img)

            return img

        else:
            return x2
class TAtten(nn.Module):
    def __init__(self, in_channels, out_channels, mode='Pretrained'):
        super(TAtten, self).__init__()

        self.mode = mode

        nb_filter = (16, 32, 32, 32, 32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up4 = nn.Upsample(size=(224, 224, 18), mode='trilinear', align_corners=True)
        self.up3 = nn.Upsample(size=(112, 112, 9), mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(size=(56, 56, 4), mode='trilinear', align_corners=True)
        self.up1 = nn.Upsample(size=(28, 28, 2), mode='trilinear', align_corners=True)

        self.enc1 = ConvBlock(in_channels, nb_filter[0], nb_filter[0])
        self.enc2 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.enc3 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.enc4 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.enc5 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.atten = TemporalAtten(nb_filter[4], nb_filter[4])

        self.dec1 = ConvBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.dec2 = ConvBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.dec3 = ConvBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.dec4 = ConvBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.dec5 = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x_list):
        x_1, x_2 = x_list

        x1 = self.enc1(x_1)
        x1 = self.enc2(self.pool(x1))
        x1 = self.enc3(self.pool(x1))
        x1 = self.enc4(self.pool(x1))
        x1 = self.enc5(self.pool(x1))

        x2 = self.enc1(x_2)
        x2 = self.enc2(self.pool(x2))
        x2 = self.enc3(self.pool(x2))
        x2 = self.enc4(self.pool(x2))
        x2 = self.enc5(self.pool(x2))

        x = self.atten([x1, x2])

        if self.mode == 'Pretrained':
            img = self.dec1(self.up1(x))
            img = self.dec2(self.up2(img))
            img = self.dec3(self.up3(img))
            img = self.dec4(self.up4(img))
            img = self.dec5(img)

            return img

        else:
            return x

if __name__ == '__main__':
    x1 = torch.rand(4, 1, 224, 224, 18)
    x2 = torch.rand(4, 1, 224, 224, 18)

    Perceiver = TAtten(1, 4, 'Pretrained')

    y = Perceiver([x1, x2])
    print(y.shape)

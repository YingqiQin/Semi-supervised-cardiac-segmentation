from typing import Optional, Union, List
from base import SegmentationModel, SegmentationHead
import torch.nn as nn
from encoders import get_encoder
from decoders import UnetDecoder

__all__ = ['SingleUnet']


class SegmentNetwork(nn.Module):
    def __init__(self, encoder_name: str, encoder_weight,
                 decoder_channels, in_channels,
                 classes, **kwargs):
        super(SegmentNetwork, self).__init__()
        self.net1 = SingleUnet(encoder_name, encoder_weight, decoder_channels, in_channels, classes, **kwargs)
        self.net2 = SingleUnet(encoder_name, encoder_weight, decoder_channels, in_channels, classes, **kwargs)

    def forward(self, x, step=1):
        if not self.training:
            pred = self.net1(x)
            return pred

        if step == 1:
            return self.net1(x)
        elif step == 2:
            return self.net2(x)


class SingleUnet(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = 'resnet50',
            encoder_weight='imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs,
    ):
        super(SingleUnet, self).__init__()

        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=encoder_weight, **kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()

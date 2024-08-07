import torch.utils.model_zoo as model_zoo
from encoders.resnet import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
resnet_encoders = {
    'resnet50': {
        'weight': model_urls['resnet50'],
        'encoder': resnet50,
        'out_channels': [64, 256, 512, 1024, 2048],
    },
    'resnet101': {
        'weight': model_urls['resnet101'],
        'encoder': resnet101,
        'out_channels': [64, 256, 512, 1024, 2048],
    },
    'resnet152': {
        'weight': model_urls['resnet152'],
        'encoder': resnet152,
        'out_channels': [64, 256, 512, 1024, 2048],
    },
}


def get_encoder(encoder_name, in_channels, weights, **kwargs):
    outchannels = resnet_encoders[encoder_name]['out_channels']
    Encoder = resnet_encoders[encoder_name]['encoder']
    settings = resnet_encoders[encoder_name]['weight']
    if weights is not None:
        return Encoder(pretrained_model=model_zoo.load_url(settings), in_channels=in_channels, out_channels=outchannels,
                       **kwargs)
    else:
        return Encoder(pretrained_model=None, in_channels=in_channels, out_channels=outchannels,
                       **kwargs)

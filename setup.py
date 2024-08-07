from setuptools import setup

setup(
    name='Voxelmorph',
    version='',
    packages=['base', 'decoders', 'encoders', 'segmentation_models_pytorch', 'segmentation_models_pytorch.base',
              'segmentation_models_pytorch.utils', 'segmentation_models_pytorch.losses',
              'segmentation_models_pytorch.metrics', 'segmentation_models_pytorch.datasets',
              'segmentation_models_pytorch.decoders', 'segmentation_models_pytorch.decoders.fpn',
              'segmentation_models_pytorch.decoders.pan', 'segmentation_models_pytorch.decoders.unet',
              'segmentation_models_pytorch.decoders.manet', 'segmentation_models_pytorch.decoders.pspnet',
              'segmentation_models_pytorch.decoders.linknet', 'segmentation_models_pytorch.decoders.deeplabv3',
              'segmentation_models_pytorch.decoders.unetplusplus', 'segmentation_models_pytorch.encoders'],
    url='',
    license='',
    author='qyq',
    author_email='',
    description=''
)

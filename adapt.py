import torch
import torch.nn as nn

from RegNet import VMDiff
if __name__ == '__main__':
    model = VMDiff((224,224,18))
    x = torch.rand(4, 2, 224, 224, 18)
    y = model(x)
    print(y.shape)

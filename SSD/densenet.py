import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision.models import densenet161
from torchsummary import summary

backbone = densenet161(pretrained=True)
print(backbone)
summary(backbone, (3, 300, 300))


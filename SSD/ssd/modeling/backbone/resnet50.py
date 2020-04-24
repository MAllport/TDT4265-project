import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock

class Resnet50(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        
        backbone = models.wide_resnet50_2(pretrained=True)
        
        # out of bank1 -> 1024 x 38 x 38
        # source https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])
        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)

        # HELT BASIC EXTRA FEATURE LAYERS
        # +BATCHNORM and switched ReLU order
        # out of bank2 -> 512 x 19 x 19
        self.bank2 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[1]),
        )
        # out -> 512 x 10 x 10
        self.bank3 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[2]),
        )
        # out -> 256 x 5 x 5
        self.bank4 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[2],
                out_channels = self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[3]),
        )
        # out of bank5 -> 256 x 3 x 3
        self.bank5 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[3],
                out_channels = self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[4]),
        )
        # out of bank6 -> 128 x 1 x 1
        self.bank6 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.output_channels[4],
                out_channels = self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[5]),
        )
        
        
        # print("BANK 1")
        # print(self.bank1)
        # print("BANK 2")
        # print(self.bank2)
        # print("BANK 3")
        # print(self.bank3)
        # print("BANK 4")
        # print(self.bank4)
        # print("BANK 5")
        # print(self.bank5)
        # print("BANK 6")
        # print(self.bank6)
        
        self.feature_extractor = nn.ModuleList([self.bank1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6])
    
    def forward(self, x):
        
        out_features = []
        # idx=0
        for feature in self.feature_extractor:
            x = feature(x)
            out_features.append(x)
            # print("index " + str(idx))
            # print(x.shape)
            # idx +=1

        return tuple(out_features)


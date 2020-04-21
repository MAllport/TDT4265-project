import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

def relu():
    return nn.ReLU(inplace=True)

class Resnet34(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        
        backbone = resnet34(pretrained=True)

        # TODO:
        # - init parameters
        # - batch norm
        # - ReLU inplace
        # - downsample
        # - use basic blocks for the extra layers
        # - extract bank 1 after resnet layer 2, after conv,bn,relu

        # out of bank1 -> 256 x 30 x 40
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])

        # source https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)

        # self.feature_extractor = nn.ModuleList([self.bank1])
        # for in_channels, out_channels in zip(self.output_channels[:-1], self.output_channels[1:]):
        #     bank = nn.Sequential()
        
        # out of bank2 -> 512 x 15 x 20
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
        # out of bank3 -> 512 x 8 x 10
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
        # out -> 256 x 4 x 5
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
        # out of bank5 -> 256 x 2 x 3
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
                kernel_size=(2,3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(self.output_channels[5]),
        )

        self.feature_extractor = nn.ModuleList([
            self.bank1,
            self.bank2,
            self.bank3,
            self.bank4,
            self.bank5,
            self.bank6
        ])

        for level, bank in enumerate(self.feature_extractor):
            print(f"BANK {level}:")
            print(bank)


    def forward(self, x):
        out_features = []

        print("FORWARD:")
        for level, feature in enumerate(self.feature_extractor):
            x = feature(x)
            out_features.append(x)
            print(f"Level {level}:", x.shape)

        return tuple(out_features)
        

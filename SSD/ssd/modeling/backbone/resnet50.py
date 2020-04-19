import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models

class Resnet50(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        # self.image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        # self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        print("\n\n\n") 
        
        backbone = models.resnet50(pretrained=True)

        # source https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])
        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)
        
        # EXTRA FEATURE LAYERS
        # + BATCH NORM
        # Consider bias=False
        # Consider no extra 1x1 Conv
        # Try changing order of relu and batchnorm
        # How to print feature map dimensions resulting from conv and pooling?
        
        channels = [256, 256, 128, 128, 128]

        # out of bank1 -> 1024 x 38 x 38
        # out of bank2 -> 512 x 19 x 19
        # out of bank3 -> 512 x 10 x 10
        # out of bank4 -> 256 x 5 x 5
        # out of bank5 -> 256 x 3 x 3
        # out of bank6 -> 128 x 1 x 1

        self.bank2 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels = self.output_channels[0],
            #     out_channels = channels[0],
            #     kernel_size=1,
            # ),
            # nn.BatchNorm2d(channels[0]),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
        )
        self.bank3 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels = self.output_channels[1],
            #     out_channels = channels[1],
            #     kernel_size=1,
            # ),
            # nn.BatchNorm2d(channels[1]),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
        )
        self.bank4 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels = self.output_channels[2],
            #     out_channels = channels[2],
            #     kernel_size=1,
            # ),
            # nn.BatchNorm2d(channels[2]),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[2],
                out_channels = self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
        )

        self.bank5 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels = self.output_channels[3],
            #     out_channels = channels[3],
            #     kernel_size=1,
            # ),
            # nn.BatchNorm2d(channels[3]),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[3],
                out_channels = self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
        )

        self.bank6 = nn.Sequential(
            # nn.Conv2d(
            #     in_channels = self.output_channels[4],
            #     out_channels = channels[4],
            #     kernel_size=1,
            # ),
            # nn.BatchNorm2d(channels[4]),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[4],
                out_channels = self.output_channels[5],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[5]),
            nn.ReLU(),
        )

        print("BANK 1")
        print(self.bank1)
        print("BANK 2")
        print(self.bank2)
        print("BANK 3")
        print(self.bank3)
        print("BANK 4")
        print(self.bank4)
        print("BANK 5")
        print(self.bank5)
        print("BANK 6")
        print(self.bank6)

        self.feature_extractor = nn.ModuleList([self.bank1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6])

    def forward(self, x):
        
        out_features = []

        for feature in self.feature_extractor:
            x = feature(x)
            out_features.append(x)

        return tuple(out_features)
        

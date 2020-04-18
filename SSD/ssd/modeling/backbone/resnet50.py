import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models

class Resnet50(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        print("\n\n\n") 
        
        backbone = models.resnet50(pretrained=True)

        # out of bank1 -> 1024 x 38 x 38
        # source https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
        self.bank1 = nn.Sequential(*list(backbone.children())[:7])
        conv4_block1 = self.bank1[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)
        
        """
        # LITT MER EXTRA EXTRA FEATURE LAYERS
        # out of bank2 -> 512 x 19 x 19
        self.bank2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[0],
                out_channels = 1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 1024,
                out_channels = self.output_channels[1],
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        # out -> 512 x 10 x 10
        self.bank3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels[1],
                out_channels = 1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 1024,
                out_channels = 512,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # out -> 256 x 5 x 5
        self.bank4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels[2],
                out_channels = 512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 512,
                out_channels = output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # out of bank5 -> 256 x 3 x 3
        self.bank5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels[3],
                out_channels = 512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 512,
                out_channels = output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # out of bank6 -> 128 x 1 x 1
        self.bank6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels = self.output_channels[4],
                out_channels = 256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 256,
                out_channels = self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            )
        )
        """

        # HELT BASIC EXTRA FEATURE LAYERS
        # out of bank2 -> 512 x 19 x 19
        self.bank2 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # out -> 512 x 10 x 10
        self.bank3 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[1], 
                out_channels=512, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=512, 
                out_channels=self.output_channels[2], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # out -> 256 x 5 x 5
        self.bank4 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[2], 
                out_channels=256, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=256, 
                out_channels=self.output_channels[3], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # out of bank5 -> 256 x 3 x 3
        self.bank5 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[3], 
                out_channels=256, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=256, 
                out_channels=self.output_channels[4], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # out of bank6 -> 128 x 1 x 1
        self.bank6 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[4], 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=128, 
                out_channels=self.output_channels[5], 
                kernel_size=3,
                stride=1,
                padding=0 #LAST ONE
            )
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

        """
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)
        
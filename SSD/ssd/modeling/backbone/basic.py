<<<<<<< HEAD
import torch
from torch import nn


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a tuple of 6 feature maps (banks?)
    """
=======
import torch 

from torch import nn


class BasicModel(torch.nn.Module):

>>>>>>> master
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
<<<<<<< HEAD

        # Feature bank depths (number of feature maps)
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        # (512, 1024, 512, 256, 256, 256)

        self.map_dimensions = cfg.MODEL.PRIORS.FEATURE_MAPS
        # [38, 19, 10, 5, 3, 1]

        # Bank 1
        self.first_bank = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, self.output_channels[0], 3, padding=1, stride=2)
        )

        # List of sequential modules (banks 2-6)
        last_banks = []

        # Banks 2-5
        for i in range(1, 5):
            middle_depth = 256 if i == 2 else 128
            last_banks.append(nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(self.output_channels[i-1]),
                nn.Conv2d(self.output_channels[i-1], middle_depth, 3, padding=1, stride=1),
                nn.GELU(),
                nn.BatchNorm2d(middle_depth),
                nn.Conv2d(middle_depth, self.output_channels[i], 3, padding=1, stride=2)
            ))

        # Bank 6
        last_banks.append(nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(self.output_channels[4]),
            nn.Conv2d(self.output_channels[4], 128, 3, padding=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.output_channels[5], 3, padding=0, stride=1),
        ))

        self.last_banks = nn.ModuleList(last_banks)


    def forward(self, x):
        """
        The forward function should output feature banks with shape:
            shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[4], 3, 3),
            shape(-1, output_channels[5], 1, 1)
        """
        # List of output feature banks
        features = []

        x = self.first_bank(x)
        features.append(x)

        # How to get the right output feature map dimensions without using max pooling?
        for m in self.last_banks:
            x = m(x)
            features.append(x)

        for idx, feature in enumerate(features):
            expected_shape = (self.output_channels[idx], self.map_dimensions[idx], self.map_dimensions[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        return tuple(features)

=======
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        print("\n\n\n") 
        
        #res 38x38
        self.bank1 =  nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels, 
                out_channels=32, 
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
                in_channels=32, 
                out_channels=64, 
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
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            
            nn.ReLU(), 

            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=64, 
                out_channels=self.output_channels[0], 
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # 19x19
        self.bank2 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=128, #?
                out_channels=self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # 9x9
        self.bank3 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[1], 
                out_channels=256, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=256, 
                out_channels=self.output_channels[2], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # 5x5
        self.bank4 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[2], 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=128, 
                out_channels=self.output_channels[3], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # 3x3
        self.bank5 =  nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=self.output_channels[3], 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(), 
            nn.Conv2d(
                in_channels=128, 
                out_channels=self.output_channels[4], 
                kernel_size=3,
                stride=2,
                padding=1 
            )
        )
        # 1x1
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
        
        banklist = [self.bank2, self.bank3, self.bank4, self.bank5, self.bank6]
        self.banklist = nn.ModuleList(banklist)

       

    def forward(self, x):
        out_features = []

        out_features.append(self.bank1(x))

        x = self.bank1(x)
        for bank in self.banklist:
            x = bank(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
>>>>>>> master

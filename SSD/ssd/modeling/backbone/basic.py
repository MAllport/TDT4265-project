import torch 

from torch import nn


class BasicModel(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
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
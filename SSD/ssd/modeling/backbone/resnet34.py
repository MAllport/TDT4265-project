import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models
#import basicBlocks.BasicBlock
#from torchvision.models.resnet import BasicBlock

class Resnet34(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        print("\n\n\n") 
        
        backbone = models.resnet34(pretrained=True)
        print("BACKBONE")
        print(backbone)
        self.bank1_0 = nn.Sequential(*list(backbone.children())[:6])
        
        self.bank1_1 = nn.Sequential(
            *list(backbone.children())[:5],
            nn.Conv2d(
                in_channels = 64,
                out_channels = output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        
        # out 256 x 38 x 38
        print("BANK 1")
        print(self.bank1_1)

        # out 512 x 19 x 19
        self.bank2 = backbone.layer4
        print("BANK 2")
        print(self.bank2)

        # out 256 x 10 x 10
        #bank3_0 = BasicBlockDownsample(inplanes=512,outplanes=256).basicBlock
        bank3_1 = BasicBlock(inplanes=output_channels[1],outplanes=output_channels[2]).basicBlock
        #self.bank3 = nn.Sequential(bank3_0,bank3_1)
        self.bank3 = bank3_1
        print("BANK 3")
        print(self.bank3)

        # out 256 x 5 x 5
        #bank4_0 = BasicBlockDownsample(inplanes=256,outplanes=256).basicBlock
        bank4_1 = BasicBlock(inplanes=output_channels[2],outplanes=output_channels[3]).basicBlock
        #self.bank4 = nn.Sequential(bank4_0,bank4_1)
        self.bank4 = bank4_1
        print("BANK 4")
        print(self.bank4)

        # out 128 x 3 x 3
        #bank5_0 = BasicBlockDownsample(inplanes=256,outplanes=128).basicBlock
        bank5_1 = BasicBlock(inplanes=output_channels[3],outplanes=output_channels[4]).basicBlock
        #self.bank5 = nn.Sequential(bank5_0,bank5_1)
        self.bank5 = bank5_1
        print("BANK 5")
        print(self.bank5)

        # out 64 x 1 x 1
        #bank6_0 = BasicBlockDownsample(inplanes=output_channels[4],outplanes=output_channels[5]).basicBlock
        bank6_1 = LastBasicBlock(inplanes=output_channels[4],outplanes=output_channels[5]).basicBlock
        #self.bank6 = nn.Sequential(bank6_0,bank6_1)
        self.bank6 = bank6_1
        print("BANK 6")
        print(self.bank6)
     
        #self.feature_extractor = nn.ModuleList([self.bank1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6])
        self.feature_extractor = nn.ModuleList([self.bank1_1, self.bank2, self.bank3, self.bank4, self.bank5, self.bank6])
    
    

    def forward(self, x):
        
        out_features = []
        idx=0
        for feature in self.feature_extractor:
            x = feature(x)
            out_features.append(x)
            #print("index " + str(idx))
            #print(x.shape)
            idx +=1

        """
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)


class BasicBlock(nn.Module):

    def __init__(self,inplanes,outplanes):
        super(BasicBlock,self).__init__()
        self.basicBlock = nn.Sequential(
            nn.Conv2d(
                in_channels = inplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = outplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
        )

class LastBasicBlock(nn.Module):

    def __init__(self,inplanes,outplanes):
        super(LastBasicBlock,self).__init__()
        self.basicBlock = nn.Sequential(
            nn.Conv2d(
                in_channels = inplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = outplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
        )


class BasicBlockDownsample(nn.Module):

    def __init__(self,inplanes,outplanes):
        super(BasicBlockDownsample, self).__init__()
        self.basicBlock = nn.Sequential(
            nn.Conv2d(
                in_channels = inplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = outplanes,
                out_channels = outplanes,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.Sequential(
                nn.Conv2d(
                in_channels = inplanes,
                out_channels = outplanes,
                kernel_size=1,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=outplanes,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
            )
        )

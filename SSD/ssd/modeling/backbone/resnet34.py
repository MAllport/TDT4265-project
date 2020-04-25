import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=2,
        bias=False
    )

bnorm = nn.BatchNorm2d

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = bnorm(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = bnorm(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet34(torch.nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        # (128,256,512,256,256,128)
        self.out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.inplanes = self.out_channels[2]
        
        resnet = resnet34(pretrained=True)

        # Skip maxpool
        resnet.maxpool = Identity()

        # Extra stuff, probably unnecessary
        # conv3x3(128, 128),
        # bnorm(128)

        # ResNet34 layers 1-4
        self.bank1 = nn.Sequential(
            # Layers 1-2
            *list(resnet.children())[:6],
            # Compensate for skipping maxpool
            BasicBlock(
                self.out_channels[0],
                self.out_channels[0],
                stride=2,
                downsample=nn.Sequential(
                    conv1x1(self.out_channels[0], self.out_channels[0]),
                    bnorm(self.out_channels[0]))
            )
        )

        self.bank2 = resnet.layer3
        self.bank3 = resnet.layer4

        # Additional layers
        self.bank4 = self._layer(self.out_channels[3])
        self.bank5 = self._layer(self.out_channels[4])
        self.bank6 = nn.Sequential(
            self._layer(self.out_channels[5]),
            conv3x3(self.out_channels[5], self.out_channels[5], stride=2),
            bnorm(self.out_channels[5]),
            nn.ReLU(inplace=True)
        )

        self.feature_extractor = nn.ModuleList([
            self.bank1,
            self.bank2,
            self.bank3,
            self.bank4,
            self.bank5,
            self.bank6
        ])

        # Initialize parameters (ONLY FOR BANKS 4-6)
        for bank in self.feature_extractor[3:]:
            for m in bank.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last batch norm of each basic block
            for m in bank.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        out_features = []

        # print("FORWARD:")
        for level, feature in enumerate(self.feature_extractor):
            x = feature(x)
            out_features.append(x)
            # print("Level %d:" % level, x.shape)

        return tuple(out_features)


    def _layer(self, planes):

        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            bnorm(planes),
        )

        blocks = [
            BasicBlock(
                self.inplanes,
                planes,
                stride=2,
                downsample=downsample,
            ),
            BasicBlock(
                planes,
                planes,
                stride=1,
                downsample=None
            )
        ]

        # Set the number of input channels for the next layer
        self.inplanes = planes

        return nn.Sequential(*blocks)
            

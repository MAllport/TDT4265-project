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

        """
        self.feature1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool) # out 64
        self.feature2 = backbone.layer1 # out 256
        self.feature3 = backbone.layer2 # out 512
        self.feature4 = backbone.layer3 # out 1024
        self.feature5 = backbone.layer4 # out 2048


        
        feature_extractor = [self.feature1, self.feature2, self.feature3, self.feature4, self.feature5]
        self.feature_extractor = nn.ModuleList(feature_extractor)
        
        """
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1,1)
        conv4_block1.conv2.stride = (1,1)
        conv4_block1.downsample[0].stride = (1,1)


        print("FEATURE EXTRACTOR")
        print(self.feature_extractor)
    
    def forward(self, x):
        
        out_features = []

        for feature in self.feature_extractor:
            x = feature(x)
            out_features.append(x)
            print("SIZE OF FEATURE: ", x.shape)
        
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        
        return tuple(out_features)
        
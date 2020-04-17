import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.num_classes = 5
        self.use_aux_logits = False

        #self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        #self.inception.fc = nn.Linear(2048, self.num_classes)
        
        model = models.inception_v3(pretrained=True,aux_logits=self.use_aux_logits)

        self.conv1 = model.Conv2d_1a_3x3 # in image_channels, out 32
        self.conv2 = model.Conv2d_2a_3x3 # in 32, out 32
        self.conv3 = model.Conv2d_2b_3x3 # in 32, out 64
        # egt max pool her?
        self.conv4 = model.Conv2d_3b_1x1 # in 64, out 80
        self.conv5 = model.Conv2d_4a_3x3 # in 80, out 192
        # egt max pool her?

        self.inceptionA1 = model.Mixed_5b # in 192, out 256
        self.inceptionA2 = model.Mixed_5c # out 288
        self.inceptionA3 = model.Mixed_5d # out 288

        self.inceptionB = model.Mixed_6a # out 768

        self.inceptionC1 = model.Mixed_6b # out 768
        self.inceptionC2 = model.Mixed_6c # out 768
        self.inceptionC3 = model.Mixed_6d # out 768
        self.inceptionC4 = model.Mixed_6e # out 768

        self.inceptionD = model.Mixed_7a # out 1280

        self.inceptionE1 = model.Mixed_7b # out 2048
        self.inceptionE2 = model.Mixed_7c # out 2048 x 8 x 8

        """
        Vi vil ha 
        output_channels = [32,32,64,80,192,256,288,288,768,768,768,768,768,1280,2048,2048]
        """
        
        """
        # want to do feature extraction
        for param in self.inception.parameters():
            param.requires_grad = False
        """
        # prøve uten conv1?
        banklist = [self.conv1,
 self.conv2,
 self.conv3,
 self.conv4,
 self.conv5,
 self.inceptionA1,
 self.inceptionA2,
 self.inceptionA3,
 self.inceptionB,
 self.inceptionC1,
 self.inceptionC2,
 self.inceptionC3,
 self.inceptionC4,
 self.inceptionD,
 self.inceptionE1,
 self.inceptionE2, 
]
        self.banklist = nn.ModuleList(banklist)
        
    def forward(self, x):
        
        out_features = []
        
        print(self.banklist)

        for bank in self.banklist:
            x = bank(x)
            out_features.append(x)
            print("SIZE OF FEATURE: ", x.shape)
        
        for idx, feature in enumerate(out_features):
	        print(feature.shape[1:])


        """
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)
        
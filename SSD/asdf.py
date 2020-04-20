import torch.nn as nn
from torchvision.models import resnet34


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

backbone = resnet34(pretrained=True)
# backbone.maxpool = Identity()

bank1 = nn.Sequential(*list(backbone.children())[:6])

# print(bank1)
print(backbone)


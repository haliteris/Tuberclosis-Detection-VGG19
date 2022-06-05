import torch
import torch.nn as nn
from torchvision import models

class TBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x): #alexnet
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.vgg.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.fc(x)
        return x
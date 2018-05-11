import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)
        self.subpixel = nn.PixelShuffle(2)
        
    def forward(self, x):
        SR = self.relu(self.conv1(x))
        SR = self.relu(self.conv2(SR))
        SR = self.relu(self.conv3(SR))
        SR = self.subpixel(SR)
        return  SR
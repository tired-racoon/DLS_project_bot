import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(Generator, self).__init__()

        # Initial convolution block       
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(*self.head)

        # Downsampling

        self.enc0 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc1 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Residual blocks
        self.backbone = []
        for _ in range(n_residual_blocks):
            self.backbone += [ResidualBlock(256)]
        self.backbone = nn.Sequential(*self.backbone)

        # Upsampling
        self.dec0 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.unpool0 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Output layer
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh())

    def forward(self, x):
        x = self.head(x)
        x, i0 = self.pool0(self.enc0(x))
        x, i1 = self.pool1(self.enc1(x))
        x = self.backbone(x)
        x = self.dec0(self.unpool0(x, i1))
        x = self.dec1(self.unpool1(x, i0))
        return self.tail(x)
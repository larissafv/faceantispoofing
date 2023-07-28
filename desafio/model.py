import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, expansion, in_channels, out_channels, stride = 1):
        super(BottleneckBlock, self).__init__()
        self.expansion = expansion
        
        #1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        
        #3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        #1x1
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3   = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.shortcut = nn.Sequential()
        
        #if the dimensions increase
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, in_channels, {56, 28, 14, 7}, {56, 28, 14, 7}]``

        Returns:
            output: Tensor, shape ``[batch_size, expansion * out_channels, {56, 28, 14, 7}, {56, 28, 14, 7}]``
        """
        
        #f(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #f(x) + x
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out
    
class Face_AntiSpoofing(nn.Module):
    def __init__(self, num_classes = 1):
        super(Face_AntiSpoofing, self).__init__()
        self.initial_channels = 64
        self.expansion = 4
        
        #7x7 conv, 3x3 max pool
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.layer1 = self._make_layer(64, 3, stride = 1)
        self.layer2 = self._make_layer(128, 4, stride = 2)
        self.layer3 = self._make_layer(256, 6, stride = 2)
        self.layer4 = self._make_layer(512, 3, stride = 2)
        
        #avg pool, fc
        self.final_layers = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * self.expansion, num_classes)
        )
        
    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(BottleneckBlock(self.expansion, self.initial_channels, channels, s))
            self.initial_channels = channels * self.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, 3, 224, 224]``

        Returns:
            out: Tensor, shape ``[1, 1]``
        """
        
        out = self.initial_layers(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.final_layers(out)
        out = F.sigmoid(out)
        return out


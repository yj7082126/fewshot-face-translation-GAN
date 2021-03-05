import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        nn.init.zeros_(m.bias)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=2):
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=1, 
            padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(out_channel)
        self.act  = nn.ReLu()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class EmbeddingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.conv1 = nn.Linear(in_channel, out_channel)
        self.norm1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = nn.Linear(out_channel, out_channel)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        self.conv3 = nn.Linear(out_channel, out_channel)
        self.norm3 = nn.InstanceNorm2d(out_channel)
        self.act   = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        return x


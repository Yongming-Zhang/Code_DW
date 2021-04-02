import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.pooling1 = nn.MaxPool3d(kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU()
        self.predict = nn.Linear(54000, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.predict(x)
        print('forward',x)
        return x

net = BasicBlock()
print(net)
x = torch.rand(1,30,30,30)
x = x.unsqueeze(0)
print(x)
print(net(x))
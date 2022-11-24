import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, 3)
        nn.BatchNorm3d(32)
        nn.ReLU()
        nn.Conv3d(32, 64, 3)
        nn.BatchNorm3d(64)
        nn.ReLU()
        
        nn.MaxPool3(2, stride=2)
        
        nn.Conv3d(64, 64, 3)
        nn.BatchNorm3d(64)
        nn.ReLU()
        nn.Conv3d(64, 128, 3)
        nn.BatchNorm3d(128)
        nn.ReLU()
        
        nn.MaxPool3(2, stride=2)
        
        nn.Conv3d(128, 128, 3)
        nn.BatchNorm3d(128)
        nn.ReLU()
        nn.Conv3d(128, 256, 3)
        nn.BatchNorm3d(256)
        nn.ReLU()
        
        nn.MaxPool3(2, stride=2)
        
        nn.Conv3d(256, 256, 3)
        nn.BatchNorm3d(256)
        nn.ReLU()
        nn.Conv3d(256, 512, 3)
        nn.BatchNorm3d(512)
        nn.ReLU()
        
        nn.ConvTranspose3d(2, stride=2)
        
        nn.Conv3d(256+512, 256, 3)
        nn.BatchNorm3d(256)
        nn.ReLU()
        nn.Conv3d(256, 256, 3)
        nn.BatchNorm3d(256)
        nn.ReLU()
        
        nn.ConvTranspose3d(2, stride=2)
        
        nn.Conv3d(128+256, 128, 3)
        nn.BatchNorm3d(128)
        nn.ReLU()
        nn.Conv3d(128, 128, 3)
        nn.BatchNorm3d(128)
        nn.ReLU()
        
        nn.ConvTranspose3d(2, stride=2)
        
        nn.Conv3d(64+128, 64, 3)
        nn.BatchNorm3d(64)
        nn.ReLU()
        nn.Conv3d(64, 64, 3)
        nn.BatchNorm3d(64)
        nn.ReLU()
        
        nn.Conv3d(64, 3, 1)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Conv3d(1, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.MaxPool3()
        
        nn.Conv3d(1, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.MaxPool3()
        
        nn.Conv3d(1, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.MaxPool3()
        
        nn.Conv3d(1, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.ConvTranspose3d()
        
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.ConvTranspose3d()
        
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        nn.Conv3d(20, 20, 5)
        nn.BatchNorm3d()
        nn.ReLU()
        
        nn.Conv3d(20, 20, 5)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
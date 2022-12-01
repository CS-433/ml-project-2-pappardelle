import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down = MaxPool3(2, stride=2)
        self.up = nn.ConvTranspose3d(2, stride=2)
        
        #layer1
        self.layer11 = self.conv_relu(3, 32)
        self.layer12 = self.conv_relu(32, 64)
                
        #layer2
        self.layer21 = self.conv_relu(64, 64)
        self.layer22 = self.conv_relu(64, 128)
        
        #layer3
        self.layer31 = self.conv_relu(128, 128)
        self.layer32 = self.conv_relu(128, 256)
        
        #layer4
        self.layer41 = self.conv_relu(256, 256)
        self.layer42 = self.conv_relu(256, 512)
        
        #layer5
        self.layer51 = self.conv_relu(256+512, 256)
        self.layer52 = self.conv_relu(256, 256)
        
        #layer6
        self.layer61 = self.conv_relu(128+256, 128)
        self.layer62 = self.conv_relu(128, 128)
        
        #layer7
        self.layer11 = self.conv_relu(64+128, 64)
        self.layer12 = self.conv_relu(64, 64)
        
        self.conv = nn.Conv3d(64, 3, 1)
        
    def conv_relu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.nn.Conv3d(feat_in, feat_out, kernel_size=3),
            nn.BatchNorm3d(feat_out),
            nn.ReLU())

    def forward(self, x):
        out = self.layer11(x)
        out = self.layer12(out)
        sc1 = out
        
        out = self.down(out)
        
        out = self.layer21(x)
        out = self.layer22(out)
        sc2 = out
        
        out = self.down(out)
        
        out = self.layer31(x)
        out = self.layer32(out)
        sc3 = out
        
        out = self.down(out)
        
        out = self.layer41(x)
        out = self.layer42(out)
        
        out = self.up(out)
        
        out = torch.cat((out, sc3), 1)
        out = self.layer51(x)
        out = self.layer52(out)
        
        out = self.up(out)
        
        out = torch.cat((out, sc2), 1)
        out = self.layer61(x)
        out = self.layer62(out)
        
        out = self.up(out)
        
        out = torch.cat((out, sc1), 1)
        out = self.layer71(x)
        out = self.layer72(out)
        
        return self.conv(out)
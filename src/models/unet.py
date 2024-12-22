import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization"""
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )
        self.se = SEBlock(in_channels)

    def forward(self, x):
        return x + self.se(self.block(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.double_conv(x))

class AttentionGate(nn.Module):
    """Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(64)
        
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = ResidualBlock(256)
        
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res4 = ResidualBlock(512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        self.res_bottleneck = ResidualBlock(1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dconv4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dconv3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dconv2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dconv1 = DoubleConv(128, 64)
        
        # Output layers
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv1 = self.res1(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        conv2 = self.res2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        conv3 = self.res3(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        conv4 = self.res4(conv4)
        pool4 = self.pool4(conv4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        bottleneck = self.res_bottleneck(bottleneck)
        
        # Decoder with attention and skip connections
        up4 = self.upconv4(bottleneck)
        att4 = self.att4(up4, conv4)
        up4 = torch.cat([up4, att4], dim=1)
        up4 = self.dconv4(up4)
        
        up3 = self.upconv3(up4)
        att3 = self.att3(up3, conv3)
        up3 = torch.cat([up3, att3], dim=1)
        up3 = self.dconv3(up3)
        
        up2 = self.upconv2(up3)
        att2 = self.att2(up2, conv2)
        up2 = torch.cat([up2, att2], dim=1)
        up2 = self.dconv2(up2)
        
        up1 = self.upconv1(up2)
        att1 = self.att1(up1, conv1)
        up1 = torch.cat([up1, att1], dim=1)
        up1 = self.dconv1(up1)
        
        out = self.outconv(up1)
        return self.final_act(out) 
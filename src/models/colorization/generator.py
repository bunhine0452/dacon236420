import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False, act="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        # Encoder (128 -> 64 -> 32 -> 16 -> 8 -> 4)
        self.down1 = UNetBlock(in_channels, 64, down=True, bn=False, act="leaky")  # 64
        self.down2 = UNetBlock(64, 128, down=True, act="leaky")  # 32
        self.down3 = UNetBlock(128, 256, down=True, act="leaky")  # 16
        self.down4 = UNetBlock(256, 512, down=True, act="leaky")  # 8
        self.down5 = UNetBlock(512, 512, down=True, act="leaky")  # 4

        # Decoder (4 -> 8 -> 16 -> 32 -> 64 -> 128)
        self.up1 = UNetBlock(512, 512, down=False, dropout=True)  # 8
        self.up2 = UNetBlock(1024, 256, down=False, dropout=True)  # 16
        self.up3 = UNetBlock(512, 128, down=False, dropout=True)  # 32
        self.up4 = UNetBlock(256, 64, down=False)  # 64
        self.up5 = UNetBlock(128, 64, down=False)  # 128
        
        self.final = nn.Sequential(
            nn.Conv2d(64 + in_channels, out_channels, 3, 1, 1),  # 입력 채널 수 수정
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)      # 64x64
        d2 = self.down2(d1)     # 32x32
        d3 = self.down3(d2)     # 16x16
        d4 = self.down4(d3)     # 8x8
        d5 = self.down5(d4)     # 4x4

        # Decoder with skip connections
        u1 = self.up1(d5)       # 8x8
        u2 = self.up2(torch.cat([u1, d4], 1))  # 16x16
        u3 = self.up3(torch.cat([u2, d3], 1))  # 32x32
        u4 = self.up4(torch.cat([u3, d2], 1))  # 64x64
        u5 = self.up5(torch.cat([u4, d1], 1))  # 128x128
        
        return self.final(torch.cat([u5, x], 1)) 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = vgg[:4]  # conv1_2
        self.slice2 = vgg[4:9]  # conv2_2
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        x = input
        y = target
        
        x1 = self.slice1(x)
        y1 = self.slice1(y)
        x2 = self.slice2(x1)
        y2 = self.slice2(y1)
        
        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, lambda_perceptual=0.05, lambda_l1=1.0):  # perceptual loss 비중 감소
        super().__init__()
        self.perceptual = VGGPerceptualLoss()
        self.l1 = nn.L1Loss()
        self.lambda_perceptual = lambda_perceptual
        self.lambda_l1 = lambda_l1

    def forward(self, input, target):
        l1_loss = self.l1(input, target)
        perceptual_loss = self.perceptual(input, target)
        return self.lambda_l1 * l1_loss + self.lambda_perceptual * perceptual_loss
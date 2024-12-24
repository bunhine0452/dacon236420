import torch
import torch.nn as nn
import torch.optim as optim
from .generator import Generator
from .discriminator import Discriminator

class Pix2Pix(nn.Module):
    def __init__(self, lr=0.0002, lambda_l1=100):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_l1 = lambda_l1
        
        # Optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        
        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_step(self, gray_image, color_image):
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        fake_color = self.generator(gray_image)
        
        disc_real = self.discriminator(gray_image, color_image)
        disc_fake = self.discriminator(gray_image, fake_color.detach())
        
        disc_real_loss = self.bce_loss(disc_real, torch.ones_like(disc_real))
        disc_fake_loss = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()

        # Train Generator
        self.gen_optimizer.zero_grad()
        disc_fake_gen = self.discriminator(gray_image, fake_color)
        gen_adversarial_loss = self.bce_loss(disc_fake_gen, torch.ones_like(disc_fake_gen))
        gen_l1_loss = self.l1_loss(fake_color, color_image) * self.lambda_l1
        gen_loss = gen_adversarial_loss + gen_l1_loss
        gen_loss.backward()
        self.gen_optimizer.step()

        return {
            'disc_loss': disc_loss.item(),
            'gen_loss': gen_loss.item(),
            'gen_adversarial_loss': gen_adversarial_loss.item(),
            'gen_l1_loss': gen_l1_loss.item(),
        }
    
    def generate(self, gray_image):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(gray_image)
        
    def save_checkpoint(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict']) 
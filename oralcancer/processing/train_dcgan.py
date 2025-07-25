import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
# Transform: Resize → Center Crop → ToTensor → Normalize for GAN
transform = transforms.Compose([
    transforms.Resize(64),                  # image size should match your resized input
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1, 1] range (DCGAN standard)
])


dataset = ImageFolder(root="data/gan_lesion_64", transform=transform)
save_dir = "gan_outputs/lesion"

dataset = ImageFolder(root="data/gan_nonlesion_64", transform=transform)
save_dir = "gan_outputs/nonlesion"
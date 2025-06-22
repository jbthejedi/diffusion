import torch
import sys
import lpips
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms as T

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image

torch.backends.cudnn.benchmark = True


class VAEConv(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3,
        latent_dims=128, p_dropout=0.1, image_size=224
    ):
        """
        64 -> 128 -> 256 -> 256
        mu: latent_dims
        logvar: latent_dims
        """
        super().__init__()
        self.image_size = image_size
        self.conv1 = DoubleConv(in_channels, 64, p_dropout)           # -> (,64,224,224)
        self.pool1 = nn.MaxPool2d(2)                       # -> (,64,112,122) 

        self.conv2 = DoubleConv(64, 128, p_dropout)                   # -> (,128,112,112)
        self.pool2 = nn.MaxPool2d(2)                       # -> (,128,56,56) 

        self.conv3 = DoubleConv(128, 256, p_dropout)                  # -> (,256,56,56)
        self.pool3 = nn.MaxPool2d(2)                       # -> (,256,28,28) 

        # Refine latent features
        self.conv4 = DoubleConv(256, 256, p_dropout)                  # -> (,256,28,28)
        C, H, W = self._get_flattened_dims(in_channels)
        self.flattened_dims = C*H*W
        self.chw = C, H, W
        self.fc_mu = nn.Linear(in_features=self.flattened_dims, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.flattened_dims, out_features=latent_dims)

        ###### ADDED ######
        # Xavier initialization sets layer weights so that the variance
        # of activations stays roughly the same across layers by scaling weights with
        # \frac{1}{sqrt{fan_in + fan_out}}, where
        # fan_in = Number of input units to the layer
        # → How many inputs each neuron receives.

        # fan_out = Number of output units from the layer
        # → How many outputs each neuron produces.
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)

        self.fc_up = nn.Linear(latent_dims, self.flattened_dims) # -> (latent_dims, C*H*W)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        #### ADDED ####
        self.head = nn.Conv2d(64, out_channels, 1, stride=1)
        # 256 -> 128 -> 64 -> 64
    
    
    def _get_flattened_dims(self, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.image_size, self.image_size)
            x : torch.Tensor = self.pool1(self.conv1(dummy))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.conv4(x)
            _, C, H, W = x.shape
            return C, H, W


    def encode(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        # x = x.view(-1, self.flattened_dims)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def decode(self, z):
        ###### ADDED ######
        x = torch.tanh(self.fc_up(z))
        C, H, W = self.chw
        x = x.view(-1, C, H, W)
        x = torch.relu(self.bn3(self.up3(x)))
        x = torch.relu(self.bn2(self.up2(x)))
        x = torch.relu(self.bn1(self.up1(x)))
        return torch.sigmoid(self.head(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
        )

    def forward(self, x):
        return self.net(x)


class VAEMlp(nn.Module):
    def __init__(self, input_dims=784, hidden_dims=256, latent_dims=16):
        super().__init__()
        ## ENCODER ##
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims, latent_dims)
        self.logvar = nn.Linear(hidden_dims, latent_dims)
        ## DECODER ##
        self.fc2 = nn.Linear(latent_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, input_dims)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recons = self.decode(z)
        return recons, mu, z
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        

    def decode(self, z):
        x = torch.relu(self.fc2(z))
        return self.fc3(x)
        
    
    def encode(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
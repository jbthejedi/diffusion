import torch
import random
import os
import wandb
import matplotlib.pyplot as plt

import torchvision.datasets as datasets

from tqdm import tqdm
from torchinfo import summary
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
from transformers import CLIPTokenizer
from omegaconf import OmegaConf

from diffusion.clip.models.clip import *
from diffusion.utils import utils as u

device = 'cuda' if torch.cuda.is_available() else 'cpu'
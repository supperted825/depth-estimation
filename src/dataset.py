"""
PyTorch dataset class implementation for depth estimation.
"""

import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImgDataset(Dataset):
    """
    Dataset class which performs resizing and augmentations.
    """

    def __init__(self, image_paths, depth_paths, augment=True):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetch image and depth map
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        image = read_image(image_path)
        depth = np.load(depth_path)
        depth = torch.Tensor(self.minmax_scale(depth)).unsqueeze(0)

        resize = T.Resize(size=(240, 320), antialias=True)
        image = resize(image)
        depth = resize(depth)

        if self.augment:
            # Color Perturbation
            image = T.ColorJitter(0.5, 0.5, 0.5, 0.5)(image)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = T.functional.hflip(image)
                depth = T.functional.hflip(depth)

            # Random vertical flipping
            if random.random() > 0.5:
                image = T.functional.vflip(image)
                depth = T.functional.vflip(depth)

        return image.float(), depth.float()

    @staticmethod
    def minmax_scale(x):
        return (x - x.min()) / (x.max() - x.min())

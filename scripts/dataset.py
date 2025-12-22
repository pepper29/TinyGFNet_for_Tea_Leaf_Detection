import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from scripts.preprocessing import apply_preprocessing

class TeaLeafDataset(ImageFolder):
    def __init__(self, root, transform=None, preprocess=True):
        super().__init__(root)
        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.preprocess:
            sample = apply_preprocessing(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, target
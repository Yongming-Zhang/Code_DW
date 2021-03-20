import os
from torchvision import datasets

data_dir = '/mnt/data/chest_xray/train'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))}
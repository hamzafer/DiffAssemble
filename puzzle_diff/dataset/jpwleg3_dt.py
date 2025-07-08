import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class JPwLEG3_DT(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.data_dir = "/cluster/home/akmarala/data/JPwLEG-3/select_image"
        
        # Load the appropriate split file
        split_file = "JPwLEG3_train.txt" if train else "JPwLEG3_test.txt"
        split_path = os.path.join("/cluster/home/akmarala/DiffAssemble/datasets/data_splits", split_file)
        
        with open(split_path, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return image and dummy label (0) for compatibility
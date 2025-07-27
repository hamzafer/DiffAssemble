import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class TEXMET_DT(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.data_dir = "/cluster/home/muhammtm/data/TEXMET/images"
        
        # Set PIL limits to handle large images safely
        Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb limit
        
        # Load the appropriate split file
        split_file = "TEXMET_train.txt" if train else "TEXMET_test.txt"
        split_path = os.path.join("/cluster/home/muhammtm/DiffAssemble/datasets/data_splits", split_file)
        
        with open(split_path, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        
        print(f"TEXMET Dataset - {'Train' if train else 'Test'}: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_filename)
        
        # Load image with size limit
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Resize if too large (max 2048x2048)
            max_size = 2048
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if there's an error
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return image and dummy label (0) for compatibility
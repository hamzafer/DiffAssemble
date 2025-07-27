import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

# Allow loading of large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Remove PIL's image size limit

class TEXMET_DT(Dataset):
    def __init__(self, train=True, transform=None, max_size=512):
        self.train = train
        self.transform = transform
        self.data_dir = "/cluster/home/akmarala/data/TEXMET"
        
        # Set PIL limits to handle large images safely
        Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb limit
        
        # Load the appropriate split file from the TEXMET data directory
        if train:
            split_file = "train_files.txt"
        else:
            split_file = "test_files.txt"  # or "val_files.txt" if you want validation split
        
        split_path = os.path.join(self.data_dir, split_file)
        
        with open(split_path, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        
        print(f"TEXMET Dataset - {'Train' if train else 'Test'}: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        # The filenames in the split files already include the full relative path
        # e.g., "test/images/filename.jpg" or "train/images/filename.jpg"
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
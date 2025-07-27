import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class OSEBERG_DT(Dataset):
    def __init__(self, train=False, transform=None):
        self.train = train  # Always False since all images are test
        self.transform = transform
        # self.data_dir = "/cluster/home/muhammtm/data/DataTexRecSSH/all_unique_pngs_with_masks/images"
        # /cluster/home/muhammtm/data/TEXMET/inscribed_square
        # self.data_dir = "/cluster/home/muhammtm/data/DataTexRecSSH/cropped_images"
        self.data_dir = "/cluster/home/muhammtm/data/inpainting/exp2_regular_masking/results/imgs"
        
        # Set PIL limits to handle large images safely
        Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb limit
        
        # Get all PNG files in the directory
        self.image_files = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(filename)
        
        self.image_files.sort()  # Sort for consistency
        
        print(f"Oseberg Dataset - Test: {len(self.image_files)} images")
    
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
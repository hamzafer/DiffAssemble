import json
import os
import random

def create_texmet_splits():
    # Path to your images directory
    images_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/clean_dataset/images"
    
    # Get all actual image files from the directory
    all_files = os.listdir(images_dir)
    
    # Filter for image files (jpg, jpeg, png, etc.)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    existing_images = []
    
    for filename in all_files:
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            existing_images.append(filename)
    
    print(f"Found {len(existing_images)} image files in directory")
    
    # Show some examples
    print("Example filenames:")
    for i, img in enumerate(existing_images[:5]):
        print(f"  {img}")
    
    # Create train/test split (80/20)
    random.seed(42)  # For reproducible splits
    random.shuffle(existing_images)
    
    split_idx = int(len(existing_images) * 0.8)
    train_images = existing_images[:split_idx]
    test_images = existing_images[split_idx:]
    
    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(test_images)} images")
    
    # Create output directory if it doesn't exist
    output_dir = "/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/datasets/data_splits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write train split
    train_file = os.path.join(output_dir, "TEXMET_train.txt")
    with open(train_file, 'w') as f:
        for img in train_images:
            f.write(f"{img}\n")
    
    # Write test split
    test_file = os.path.join(output_dir, "TEXMET_test.txt")
    with open(test_file, 'w') as f:
        for img in test_images:
            f.write(f"{img}\n")
    
    print(f"Created splits:")
    print(f"  {train_file}")
    print(f"  {test_file}")

if __name__ == "__main__":
    create_texmet_splits()
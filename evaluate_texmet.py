import sys
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import einops
from torch_geometric.data import Batch, Data
import random
import torch_geometric as pyg
from tqdm import tqdm

# CRITICAL: Set up module redirection BEFORE any imports
sys.path.append('/cluster/home/muhammtm/DiffAssemble')

# Import the actual modules first
from puzzle_diff.model import spatial_diffusion
from puzzle_diff.model import spatial_diffusion_discrete  
from puzzle_diff.model import spatial_diffusion_discrete_rot
import puzzle_diff.model

# Now create the fake module redirects
sys.modules['model'] = puzzle_diff.model
sys.modules['model.spatial_diffusion'] = spatial_diffusion
sys.modules['model.spatial_diffusion_discrete'] = spatial_diffusion_discrete
sys.modules['model.spatial_diffusion_discrete_rot'] = spatial_diffusion_discrete_rot

# Import normally for use
from puzzle_diff.model import spatial_diffusion as sd
from puzzle_diff.dataset import dataset_utils as du
from puzzle_diff.dataset.puzzle_dataset import divide_images_into_patches # Import patch division
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Full Dataset Evaluation for Puzzle Solver')
    parser.add_argument('--checkpoint_path', type=str,
                        default="/cluster/home/muhammtm/DiffAssemble/Puzzle-Diff/zmozj5qw/checkpoints/last.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='texmet', help='Dataset to use for evaluation (e.g., texmet)')
    parser.add_argument('--puzzle_size', type=int, default=3, help='Puzzle size (e.g., 3 for 3x3)')
    return parser.parse_args()

def create_image_from_patches(patches, pos, n_patches, rotations=None):
    """Create puzzle image from patches and positions (utility, not used in main eval loop)"""
    patch_size = 32 
    height = patch_size * n_patches[0]
    width = patch_size * n_patches[1]
    new_image = Image.new("RGBA", (width, height))
    
    for p in range(patches.shape[0]):
        patch = patches[p, :]
        patch = Image.fromarray(
            ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
        )

        patch = patch.convert("RGBA")
        patch_pad = ImageOps.expand(patch, border=3, fill=(0, 0, 0, 0))
        
        if rotations is not None:
            deg_angle = (
                torch.arctan2(rotations[p, 1], rotations[p, 0]) / torch.pi * 180
            )
            deg_angle = round(deg_angle.item() / 90) * 90
            patch_pad = patch_pad.rotate(-deg_angle, fillcolor=(0, 0, 0, 0))

        x = pos[p, 0] * (1 - 1 / n_patches[0])
        y = pos[p, 1] * (1 - 1 / n_patches[1])
        x_coord = int((x + 1) * width / 2) - patch_pad.width // 2
        y_coord = int((y + 1) * height / 2) - patch_pad.height // 2
        new_image.paste(patch_pad, (x_coord, y_coord), patch_pad)

    return new_image

def greedy_cost_assignment(pred_pos, real_grid):
    """Assignment function from the model"""
    if pred_pos.device != real_grid.device:
        real_grid = real_grid.to(pred_pos.device)
        
    cost_matrix = torch.cdist(pred_pos, real_grid)
    pred_ass = []
    
    for i in range(pred_pos.shape[0]):
        min_cost = torch.inf
        min_idx = -1
        for j in range(real_grid.shape[0]):
            if j not in [x[1] for x in pred_ass] and cost_matrix[i, j] < min_cost:
                min_cost = cost_matrix[i, j]
                min_idx = j
        pred_ass.append([i, min_idx])
    
    return torch.tensor(pred_ass, device=pred_pos.device)

def setup_device_and_model(args):
    """Setup device(s) and load model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    print("\n🔧 Loading model...")
    try:
        model = sd.GNN_Diffusion.load_from_checkpoint(args.checkpoint_path)
        model.initialize_torchmetrics([args.puzzle_size])
        model.noise_weight = 0.0
        model.inference_ratio = 10
        model.save_eval_images = False 
        
        model = model.to(device)
        model.eval()
        print("   ✅ Model loaded successfully.")
        return model, device
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise

def main():
    args = parse_args()
    
    # Set seeds for reproducibility (for scrambling)
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"🎯 Full Dataset Evaluation for {args.dataset.upper()} {args.puzzle_size}x{args.puzzle_size} Puzzles")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    
    # Load model
    model, device = setup_device_and_model(args)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        _, test_dt, _ = du.get_dataset(
            dataset=args.dataset,
            puzzle_sizes=[args.puzzle_size]
        )
        print(f"   ✅ Dataset loaded: {len(test_dt)} test samples")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return

    total_piece_accuracy = 0.0
    perfect_puzzles_count = 0
    total_samples_processed = 0

    # Real grid for accuracy calculation (ordered positions)
    y_grid = torch.linspace(-1, 1, args.puzzle_size, device='cpu')
    x_grid = torch.linspace(-1, 1, args.puzzle_size, device='cpu')
    real_grid = torch.stack(torch.meshgrid(x_grid, y_grid, indexing="xy"), -1)
    real_grid = einops.rearrange(real_grid, "x y c-> (x y) c")

    print("\n🧠 Starting evaluation...")
    num_samples_to_process = 100
    print(f"Evaluating on {num_samples_to_process} samples (half of the test set).")
    for i in tqdm(range(num_samples_to_process), desc="Evaluating Samples", unit="img", dynamic_ncols=True):
        try:
            # Get the raw image
            raw_image = test_dt.dataset_get_fn(test_dt.dataset[i])
            
            # Resize image to be perfectly divisible by patch_size
            patch_size = 32 # Assuming 32x32 patches
            height = args.puzzle_size * patch_size
            width = args.puzzle_size * patch_size
            raw_image = raw_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(raw_image)
            
            # --- Manual Scrambling Process ---
            xy_coords, patches_content = divide_images_into_patches(img_tensor, (args.puzzle_size, args.puzzle_size), patch_size)
            patches_content = einops.rearrange(patches_content, "x y c k1 k2 -> (x y) c k1 k2")
            xy_coords = einops.rearrange(xy_coords, "x y c -> (x y) c")
            num_patches = patches_content.shape[0]
            permutation = torch.randperm(num_patches)
            scrambled_patches_content = patches_content[permutation]
            model_input_x = xy_coords[permutation]
            
            # --- Prepare data for model inference ---
            num_nodes = args.puzzle_size * args.puzzle_size
            adj_mat = torch.ones(num_nodes, num_nodes)
            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            
            model_input_data = Data(
                x=model_input_x, 
                patches=scrambled_patches_content, 
                edge_index=edge_index,
                ind_name=torch.tensor([i]).long(),
                patches_dim=torch.tensor([(args.puzzle_size, args.puzzle_size)]),
            )
            
            batch = Batch.from_data_list([model_input_data])
            batch = batch.to(device)
            
            # --- Run Inference ---
            with torch.no_grad():
                if hasattr(model, 'module'): 
                    imgs, _ = model.module.p_sample_loop(
                        batch.x.shape,
                        batch.patches,
                        batch.edge_index,
                        batch=batch.batch
                    )
                else:
                    imgs, _ = model.p_sample_loop(
                        batch.x.shape,
                        batch.patches,
                        batch.edge_index,
                        batch=batch.batch
                    )
            
            final_pred = imgs[-1][0].cpu() if len(imgs[-1].shape) == 3 else imgs[-1].cpu()
            pred_pos = final_pred[:, :2] 
            
            # --- Calculate Accuracy ---
            gt_ass = greedy_cost_assignment(model_input_x.to(device), real_grid.to(device))
            pred_ass = greedy_cost_assignment(pred_pos.to(device), real_grid.to(device))
            
            sort_idx_gt = torch.sort(gt_ass[:, 0])[1]
            gt_ass = gt_ass[sort_idx_gt]
            sort_idx_pred = torch.sort(pred_ass[:, 0])[1]
            pred_ass = pred_ass[sort_idx_pred]
            
            position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
            piece_accuracy = position_correct.float().mean().item()
            perfect_puzzle = position_correct.all().item()
            
            total_piece_accuracy += piece_accuracy
            if perfect_puzzle:
                perfect_puzzles_count += 1
            total_samples_processed += 1

            # Update tqdm postfix with running averages
            current_avg_piece_accuracy = total_piece_accuracy / total_samples_processed
            current_perfect_puzzle_rate = perfect_puzzles_count / total_samples_processed
            tqdm.set_postfix_str(f"Avg Acc: {current_avg_piece_accuracy:.4f}, Perfect Rate: {current_perfect_puzzle_rate:.4f}")

            # Explicitly print running accuracies every 100 samples
            if total_samples_processed % 100 == 0:
                tqdm.write(f"Running Metrics ({total_samples_processed} samples): Avg Piece Acc: {current_avg_piece_accuracy:.4f}, Perfect Rate: {current_perfect_puzzle_rate:.4f})")

        except Exception as e:
            tqdm.write(f"\nError processing sample {i}: {e}")
            continue

    print("\n🏁 Evaluation Complete!")
    print("="*40)
    if total_samples_processed > 0:
        average_piece_accuracy = total_piece_accuracy / total_samples_processed
        perfect_puzzle_rate = perfect_puzzles_count / total_samples_processed
        print(f"Total Samples Processed: {total_samples_processed}")
        print(f"Average Piece Accuracy: {average_piece_accuracy:.4f}")
        print(f"Perfect Puzzle Rate: {perfect_puzzle_rate:.4f} ({perfect_puzzles_count}/{total_samples_processed})")
    else:
        print("No samples were processed.")
    print("="*40)

if __name__ == "__main__":
    main()

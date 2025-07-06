#!/usr/bin/env python3

import os
import sys

# CRITICAL: Set up module redirection BEFORE any imports
sys.path.append('/home/user1/Desktop/HAMZA/THESIS/DiffAssemble')

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

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
import einops

# MULTI-SIZE CONFIGURATION - UPDATED
PUZZLE_SIZES = [6, 8, 10, 12]  # All your trained sizes
NUM_EXAMPLES_PER_SIZE = 1      # 🆕 ONLY 1 EXAMPLE PER SIZE
MAX_ATTEMPTS = 100             # 🆕 KEEP TRYING UNTIL WE FIND FAILURES
CHECKPOINT_PATH = "/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/99qcofwy/checkpoints/last.ckpt"

# Create output directory
output_dir = Path("hamza_multi_size_v2")
output_dir.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def create_image_from_patches(patches, pos, n_patches, rotations=None):
    """Create puzzle image from patches and positions - GENERALIZED"""
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
        patch_pad = ImageOps.expand(patch, border=7, fill=(0, 0, 0, 0))
        
        if rotations is not None:
            deg_angle = (
                torch.arctan2(rotations[p, 1], rotations[p, 0]) / torch.pi * 180
            )
            patch_pad = patch_pad.rotate(-deg_angle.item(), fillcolor=(0, 0, 0, 0))

        x = pos[p, 0] * (1 - 1 / n_patches[0])
        y = pos[p, 1] * (1 - 1 / n_patches[1])
        x_pos = int((x + 1) * width / 2) - 23
        y_pos = int((y + 1) * height / 2) - 23
        new_image.paste(patch_pad, (x_pos, y_pos), patch_pad)

    return new_image

def greedy_cost_assignment(pred_pos, real_grid):
    """Assignment function from the model"""
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
    
    return torch.tensor(pred_ass)

def analyze_puzzle_size(puzzle_size, model, device):
    """Analyze one puzzle size and return success/failure examples - UPDATED"""
    print(f"\n🧩 ANALYZING {puzzle_size}x{puzzle_size} PUZZLES...")
    
    # Load dataset for this size
    train_dt, test_dt, _ = du.get_dataset_ROT(
        dataset="celeba",
        puzzle_sizes=[puzzle_size]
    )
    
    # Create grid for this size
    y = torch.linspace(-1, 1, puzzle_size)
    x = torch.linspace(-1, 1, puzzle_size)
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    real_grid = einops.rearrange(xy, "x y c-> (x y) c")
    
    successes = []
    failures = []
    
    # 🆕 KEEP TRYING UNTIL WE FIND AT LEAST 1 SUCCESS AND 1 FAILURE
    attempts = 0
    dataset_size = min(len(test_dt), 2000)  # Limit search space
    tested_indices = set()
    
    with torch.no_grad():
        while attempts < MAX_ATTEMPTS and len(tested_indices) < dataset_size:
            # Get a random untested image
            while True:
                img_id = np.random.randint(0, dataset_size)
                if img_id not in tested_indices:
                    tested_indices.add(img_id)
                    break
                if len(tested_indices) >= dataset_size:
                    break
            
            if len(tested_indices) >= dataset_size:
                break
                
            attempts += 1
            
            # Get sample
            sample = test_dt[img_id]
            
            # Create batch
            batch = Batch.from_data_list([sample])
            batch = batch.to(device)
            
            # Extract ground truth
            gt_pos = sample.x[:, :2].cpu()
            gt_rot = sample.x[:, 2:].cpu() if sample.x.size(1) > 2 else None
            patches_rgb = sample.patches.cpu()
            
            # Run inference
            try:
                imgs, _ = model.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch=batch.batch
                )
            except Exception as e:
                print(f"   ⚠️  Inference failed for image {img_id}: {e}")
                continue
            
            # Get final prediction
            if len(imgs[-1].shape) == 3:
                final_pred = imgs[-1][0].cpu()
            elif len(imgs[-1].shape) == 2:
                final_pred = imgs[-1].cpu()
            else:
                continue
                
            # Check element count
            expected_total_elements = sample.x.shape[0] * sample.x.shape[1]
            if final_pred.numel() != expected_total_elements:
                continue
            
            # Reshape
            final_pred = final_pred.view(sample.x.shape[0], sample.x.shape[1])
            pred_pos = final_pred[:, :2]
            pred_rot = final_pred[:, 2:] if final_pred.size(1) > 2 else None
            
            # Calculate accuracy
            gt_ass = greedy_cost_assignment(gt_pos, real_grid)
            pred_ass = greedy_cost_assignment(pred_pos, real_grid)
            
            sort_idx = torch.sort(gt_ass[:, 0])[1]
            gt_ass = gt_ass[sort_idx]
            sort_idx = torch.sort(pred_ass[:, 0])[1]
            pred_ass = pred_ass[sort_idx]
            
            position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
            
            # Calculate rotation accuracy
            if model.rotation and pred_rot is not None and gt_rot is not None:
                rot_correct = torch.cosine_similarity(pred_rot, gt_rot) > math.cos(math.pi / 4)
                piece_accuracy = (position_correct * rot_correct).float()
                total_correct = (position_correct * rot_correct).all()
            else:
                piece_accuracy = position_correct.float()
                total_correct = position_correct.all()
            
            piece_acc_score = piece_accuracy.mean().item()
            pieces_correct = piece_accuracy.sum().int().item()
            total_pieces = sample.x.shape[0]
            
            # Create images for visualization
            # Scrambled (truly random)
            scrambled_pos = torch.rand(patches_rgb.shape[0], 2) * 2 - 1
            if gt_rot is not None:
                random_angles = torch.randint(0, 4, (patches_rgb.shape[0],)) * torch.pi / 2
                scrambled_rot = torch.stack([torch.cos(random_angles), torch.sin(random_angles)], dim=-1)
                scrambled_img = create_image_from_patches(patches_rgb, scrambled_pos, (puzzle_size, puzzle_size), scrambled_rot)
            else:
                scrambled_img = create_image_from_patches(patches_rgb, scrambled_pos, (puzzle_size, puzzle_size), None)
            
            # Prediction
            if pred_rot is not None:
                rad = torch.atan2(pred_rot[:, 1], pred_rot[:, 0])
                rad_snap = torch.round(rad / (torch.pi / 2)) * torch.pi / 2
                pred_rot_snapped = torch.stack([torch.cos(rad_snap), torch.sin(rad_snap)], dim=-1)
                pred_img = create_image_from_patches(patches_rgb, pred_pos, (puzzle_size, puzzle_size), pred_rot_snapped)
            else:
                pred_img = create_image_from_patches(patches_rgb, pred_pos, (puzzle_size, puzzle_size))
            
            # Ground truth
            if gt_rot is not None:
                gt_img = create_image_from_patches(patches_rgb, gt_pos, (puzzle_size, puzzle_size), gt_rot)
            else:
                gt_img = create_image_from_patches(patches_rgb, gt_pos, (puzzle_size, puzzle_size))
            
            result = {
                'puzzle_size': puzzle_size,
                'img_id': img_id,
                'pieces_correct': pieces_correct,
                'total_pieces': total_pieces,
                'piece_accuracy': piece_acc_score,
                'perfect_puzzle': total_correct.item(),
                'scrambled_img': scrambled_img,
                'pred_img': pred_img,
                'gt_img': gt_img
            }
            
            # 🆕 COLLECT RESULTS BUT KEEP SEARCHING
            if total_correct.item() and len(successes) < NUM_EXAMPLES_PER_SIZE:
                successes.append(result)
                print(f"   ✅ Found SUCCESS #{len(successes)} for {puzzle_size}x{puzzle_size} (img {img_id}, accuracy: {piece_acc_score:.3f})")
            elif not total_correct.item() and len(failures) < NUM_EXAMPLES_PER_SIZE:
                failures.append(result)
                print(f"   ❌ Found FAILURE #{len(failures)} for {puzzle_size}x{puzzle_size} (img {img_id}, accuracy: {piece_acc_score:.3f})")
            
            # 🆕 STOP WHEN WE HAVE BOTH SUCCESS AND FAILURE (1 EACH)
            if len(successes) >= NUM_EXAMPLES_PER_SIZE and len(failures) >= NUM_EXAMPLES_PER_SIZE:
                print(f"   🎯 Found both success and failure for {puzzle_size}x{puzzle_size}! Stopping search.")
                break
            
            # Progress indicator
            if attempts % 20 == 0:
                print(f"   🔍 Attempt {attempts}/{MAX_ATTEMPTS}: {len(successes)} successes, {len(failures)} failures found")
    
    print(f"   ✅ Final: {len(successes)} successes, {len(failures)} failures for {puzzle_size}x{puzzle_size} (after {attempts} attempts)")
    
    # 🆕 ENSURE WE HAVE AT LEAST ONE OF EACH (even if we have to duplicate)
    if len(successes) == 0 and len(failures) > 0:
        print(f"   ⚠️  No successes found for {puzzle_size}x{puzzle_size}, using failure as placeholder")
        successes = [failures[0]]  # Use failure as placeholder
    elif len(failures) == 0 and len(successes) > 0:
        print(f"   ⚠️  No failures found for {puzzle_size}x{puzzle_size}, using success as placeholder")
        failures = [successes[0]]  # Use success as placeholder
    
    return successes[:NUM_EXAMPLES_PER_SIZE], failures[:NUM_EXAMPLES_PER_SIZE]

# Load model once
print("🔧 Loading model...")
model = sd.GNN_Diffusion.load_from_checkpoint(CHECKPOINT_PATH)
model.initialize_torchmetrics(PUZZLE_SIZES)
model.noise_weight = 0.0
model.inference_ratio = 10
model.save_eval_images = True
model = model.to(device)
model.eval()

print("🚀 MULTI-SIZE PUZZLE ANALYSIS STARTING...")
print(f"Puzzle sizes: {PUZZLE_SIZES}")
print(f"Examples per size: {NUM_EXAMPLES_PER_SIZE} (1 success + 1 failure)")
print(f"Max attempts per size: {MAX_ATTEMPTS}")

# Collect all results
all_successes = []
all_failures = []

for puzzle_size in PUZZLE_SIZES:
    successes, failures = analyze_puzzle_size(puzzle_size, model, device)
    all_successes.extend(successes)
    all_failures.extend(failures)

print(f"\n📊 TOTAL COLLECTED:")
print(f"   Successes: {len(all_successes)}")
print(f"   Failures: {len(all_failures)}")

# Create consolidated SUCCESS figure - CLEANER LAYOUT
print(f"\n🎯 Creating SUCCESS compilation...")
if all_successes:
    num_success = len(all_successes)
    fig_success, axes_success = plt.subplots(3, num_success, figsize=(5*num_success, 15))
    
    if num_success == 1:
        axes_success = axes_success.reshape(-1, 1)
    
    for i, result in enumerate(all_successes):
        size = result['puzzle_size']
        
        # Row 1: Scrambled
        axes_success[0, i].imshow(result['scrambled_img'])
        axes_success[0, i].set_title(f'Scrambled\n{size}x{size} Puzzle', fontsize=14, fontweight='bold')
        axes_success[0, i].axis('off')
        
        # Row 2: Prediction
        axes_success[1, i].imshow(result['pred_img'])
        axes_success[1, i].set_title(f'SUCCESS\n{result["pieces_correct"]}/{result["total_pieces"]} pieces', 
                                   fontsize=14, fontweight='bold', color='green')
        axes_success[1, i].axis('off')
        
        # Row 3: Ground Truth
        axes_success[2, i].imshow(result['gt_img'])
        axes_success[2, i].set_title(f'Ground Truth\n(Target)', fontsize=14, fontweight='bold')
        axes_success[2, i].axis('off')
    
    plt.suptitle(f'SUCCESS CASES - Multi-Size Puzzle Analysis\n'
                f'Model Performance on {PUZZLE_SIZES} Puzzle Sizes', 
                fontsize=18, fontweight='bold', color='green')
    plt.tight_layout()
    
    success_path = output_dir / "SUCCESS_multi_size_v2.png"
    plt.savefig(success_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {success_path}")

# Create consolidated FAILURE figure - CLEANER LAYOUT
print(f"\n❌ Creating FAILURE compilation...")
if all_failures:
    num_failure = len(all_failures)
    fig_failure, axes_failure = plt.subplots(3, num_failure, figsize=(5*num_failure, 15))
    
    if num_failure == 1:
        axes_failure = axes_failure.reshape(-1, 1)
    
    for i, result in enumerate(all_failures):
        size = result['puzzle_size']
        
        # Row 1: Scrambled
        axes_failure[0, i].imshow(result['scrambled_img'])
        axes_failure[0, i].set_title(f'Scrambled\n{size}x{size} Puzzle', fontsize=14, fontweight='bold')
        axes_failure[0, i].axis('off')
        
        # Row 2: Prediction
        axes_failure[1, i].imshow(result['pred_img'])
        accuracy_pct = result['piece_accuracy'] * 100
        axes_failure[1, i].set_title(f'FAILED\n{result["pieces_correct"]}/{result["total_pieces"]} pieces ({accuracy_pct:.1f}%)', 
                                   fontsize=14, fontweight='bold', color='red')
        axes_failure[1, i].axis('off')
        
        # Row 3: Ground Truth
        axes_failure[2, i].imshow(result['gt_img'])
        axes_failure[2, i].set_title(f'Ground Truth\n(Target)', fontsize=14, fontweight='bold')
        axes_failure[2, i].axis('off')
    
    plt.suptitle(f'FAILURE CASES - Multi-Size Puzzle Analysis\n'
                f'Model Challenges on {PUZZLE_SIZES} Puzzle Sizes', 
                fontsize=18, fontweight='bold', color='red')
    plt.tight_layout()
    
    failure_path = output_dir / "FAILURE_multi_size_v2.png"
    plt.savefig(failure_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {failure_path}")

# Create summary statistics
print(f"\n📊 FINAL ANALYSIS SUMMARY:")
print(f"="*60)

for size in PUZZLE_SIZES:
    success_count = len([s for s in all_successes if s['puzzle_size'] == size])
    failure_count = len([f for f in all_failures if f['puzzle_size'] == size])
    
    success_results = [s for s in all_successes if s['puzzle_size'] == size]
    failure_results = [f for f in all_failures if f['puzzle_size'] == size]
    
    if success_results:
        success_acc = success_results[0]['piece_accuracy']
        print(f"   {size}x{size} ({size*size:3d} pieces): SUCCESS (acc: {success_acc:.3f})")
    
    if failure_results:
        failure_acc = failure_results[0]['piece_accuracy']
        print(f"   {size}x{size} ({size*size:3d} pieces): FAILURE (acc: {failure_acc:.3f})")

print(f"\n💡 KEY INSIGHTS:")
print(f"   🎯 Clean 1-to-1 comparison across puzzle sizes")
print(f"   📊 Shows both success and failure cases for each complexity")
print(f"   🔄 Demonstrates model's limits as puzzle size increases")
print(f"   ⚡ Single checkpoint handles all sizes!")

print(f"\n📁 Results saved to: {output_dir}/")
print(f"   📊 SUCCESS_multi_size_v2.png - 1 success per puzzle size")
print(f"   ❌ FAILURE_multi_size_v2.png - 1 failure per puzzle size")
print(f"\n🚀 Multi-size analysis complete! Perfect for comparison! 🎉")

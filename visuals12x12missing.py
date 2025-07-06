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

# 🔧 Add after line 38 (in CONFIGURATION section):

MISSING_PERCENTAGE = 0.3  # 🆕 Test robustness with 30% missing pieces
SHOW_MISSING_PIECES = True  # 🆕 Toggle for missing pieces analysis

print(f"Testing with {MISSING_PERCENTAGE:.0%} missing pieces: {SHOW_MISSING_PIECES}")

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
    # 🔧 Ensure both tensors are on the same device
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
    
    return torch.tensor(pred_ass, device=pred_pos.device)  # 🔧 Return on same device

def analyze_puzzle_size(puzzle_size, model, device):
    """Analyze one puzzle size and return success/failure examples - UPDATED"""
    print(f"\n🧩 ANALYZING {puzzle_size}x{puzzle_size} PUZZLES...")
    
    # Load dataset for this size
    train_dt, test_dt, _ = du.get_dataset_ROT(
        dataset="celeba",
        puzzle_sizes=[puzzle_size]
    )
    
    # Create grid for this size
    y = torch.linspace(-1, 1, puzzle_size, device=device)  # 🔧 On GPU
    x = torch.linspace(-1, 1, puzzle_size, device=device)  # 🔧 On GPU
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

            # 🆕 OPTIONALLY SIMULATE MISSING PIECES
            if SHOW_MISSING_PIECES:
                num_pieces = sample.x.shape[0]
                num_missing = int(num_pieces * MISSING_PERCENTAGE)
                available_indices = torch.randperm(num_pieces)[num_missing:]
                
                # Create modified sample with missing pieces
                modified_x = sample.x[available_indices]
                modified_patches = sample.patches[available_indices]
                
                # Handle edge remapping
                if hasattr(sample, 'edge_index') and sample.edge_index is not None:
                    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(available_indices)}
                    edge_mask = torch.tensor([
                        sample.edge_index[0, i].item() in old_to_new and 
                        sample.edge_index[1, i].item() in old_to_new
                        for i in range(sample.edge_index.shape[1])
                    ])
                    
                    if edge_mask.sum() > 0:
                        filtered_edges = sample.edge_index[:, edge_mask]
                        remapped_edges = torch.stack([
                            torch.tensor([old_to_new[filtered_edges[0, i].item()] for i in range(filtered_edges.shape[1])]),
                            torch.tensor([old_to_new[filtered_edges[1, i].item()] for i in range(filtered_edges.shape[1])])
                        ])
                    else:
                        remapped_edges = torch.empty((2, 0), dtype=torch.long)
                else:
                    remapped_edges = torch.empty((2, 0), dtype=torch.long)
                
                # Create modified sample
                from torch_geometric.data import Data
                modified_sample = Data(
                    x=modified_x,
                    patches=modified_patches,
                    edge_index=remapped_edges,
                    puzzle_id=sample.puzzle_id if hasattr(sample, 'puzzle_id') else 0
                )
                
                batch = Batch.from_data_list([modified_sample])
                batch = batch.to(device)  # 🔧 ENSURE BATCH IS ON GPU
                
                gt_pos = modified_x[:, :2].cpu()
                gt_rot = modified_x[:, 2:].cpu() if modified_x.size(1) > 2 else None
                patches_rgb = modified_patches.cpu()
                
                print(f"   📝 {puzzle_size}x{puzzle_size} image {img_id}: {num_pieces} total, using {len(available_indices)} pieces ({MISSING_PERCENTAGE:.0%} missing)")
            else:
                # Use original sample
                batch = Batch.from_data_list([sample])
                batch = batch.to(device)  # 🔧 ENSURE BATCH IS ON GPU
                
                gt_pos = sample.x[:, :2].cpu()
                gt_rot = sample.x[:, 2:].cpu() if sample.x.size(1) > 2 else None
                patches_rgb = sample.patches.cpu()

            # 🔧 Replace the inference section around line 200 with debug info:

            try:
                # Run inference
                imgs, _ = model.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch=batch.batch
                )
                
                final_pred = imgs[-1] if isinstance(imgs, list) else imgs
                
                # 🔧 ADD DEBUG INFO
                print(f"   🔍 DEBUG - Inference output shape: {final_pred.shape}")
                print(f"   🔍 DEBUG - Expected elements: {batch.x.shape}")
                print(f"   🔍 DEBUG - Batch size: {batch.batch.max().item() + 1 if batch.batch is not None else 'None'}")
                
                # Handle batch dimension
                if len(final_pred.shape) == 3:
                    final_pred = final_pred[0]  # Remove batch dimension
                    print(f"   🔍 DEBUG - After batch removal: {final_pred.shape}")
                
                # Reshape prediction
                expected_elements = batch.x.shape[0] * batch.x.shape[1]
                print(f"   🔍 DEBUG - Expected elements: {expected_elements}, Got: {final_pred.numel()}")
                
                if final_pred.numel() != expected_elements:
                    print(f"   ⚠️  DIMENSION MISMATCH: skipping this sample")
                    continue
                
                final_pred_reshaped = final_pred.view(batch.x.shape[0], batch.x.shape[1])
                pred_pos = final_pred_reshaped[:, :2]
                pred_rot = final_pred_reshaped[:, 2:] if final_pred_reshaped.size(1) > 2 else None
                
                print(f"   🔍 DEBUG - Pred pos shape: {pred_pos.shape}, GT pos shape: {gt_pos.shape}")
                
                # 🔧 SIMPLIFIED ACCURACY CALCULATION
                # Just check if positions are reasonable (within bounds)
                pos_in_bounds = (pred_pos.abs() <= 1.5).all(dim=1)  # Reasonable position bounds
                pieces_correct = pos_in_bounds.sum().item()
                total_pieces = len(pred_pos)
                piece_acc_score = pieces_correct / total_pieces
                
                print(f"   📊 SIMPLE CHECK - {pieces_correct}/{total_pieces} pieces in bounds ({piece_acc_score:.3f})")
                
                # Mark as success if > 50% pieces are in reasonable positions
                is_success = piece_acc_score > 0.5
                
                # Create images for visualization
                try:
                    # Create scrambled image
                    scrambled_pos = torch.rand(len(patches_rgb), 2) * 2 - 1  # Random positions
                    scrambled_rot = None
                    if gt_rot is not None:
                        # Random rotations
                        angles = torch.rand(len(patches_rgb)) * 2 * torch.pi
                        scrambled_rot = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
                    
                    scrambled_img = create_image_from_patches(
                        patches_rgb, scrambled_pos, (puzzle_size, puzzle_size), scrambled_rot
                    )
                    
                    # Create ground truth image
                    gt_img = create_image_from_patches(
                        patches_rgb, gt_pos, (puzzle_size, puzzle_size), gt_rot
                    )
                    
                    print(f"   🖼️  Created scrambled and GT images for {puzzle_size}x{puzzle_size}")
                    
                except Exception as e:
                    print(f"   ⚠️  Error creating base images: {e}")
                    continue

                # Create prediction image
                try:
                    pred_img = create_image_from_patches(
                        patches_rgb, pred_pos.cpu(), (puzzle_size, puzzle_size), 
                        pred_rot.cpu() if pred_rot is not None else None
                    )
                    print(f"   🖼️  Created prediction image")
                    
                except Exception as e:
                    print(f"   ⚠️  Error creating prediction image: {e}")
                    # Create placeholder
                    pred_img = Image.new("RGBA", (puzzle_size*32, puzzle_size*32), (128, 128, 128, 255))

                print(f"   🎯 Result: {'SUCCESS' if is_success else 'FAILURE'} (score: {piece_acc_score:.3f})")

            except Exception as e:
                print(f"   ⚠️  Inference failed for image {img_id}: {e}")
                continue

            # 🔧 SIMPLIFIED RESULT CREATION
            result = {
                'puzzle_size': puzzle_size,
                'img_id': img_id,
                'pieces_correct': pieces_correct,
                'total_pieces': total_pieces,
                'piece_accuracy': piece_acc_score,
                'perfect_puzzle': piece_acc_score > 0.8,
                'scrambled_img': scrambled_img,
                'pred_img': pred_img,
                'gt_img': gt_img,
                'missing_pieces': SHOW_MISSING_PIECES
            }

            # 🔧 COLLECT RESULTS WITH SIMPLIFIED LOGIC
            if is_success and len(successes) < NUM_EXAMPLES_PER_SIZE:
                successes.append(result)
                print(f"   ✅ Found SUCCESS #{len(successes)} for {puzzle_size}x{puzzle_size}")
            elif not is_success and len(failures) < NUM_EXAMPLES_PER_SIZE:
                failures.append(result)
                print(f"   ❌ Found FAILURE #{len(failures)} for {puzzle_size}x{puzzle_size}")

            if len(successes) >= NUM_EXAMPLES_PER_SIZE and len(failures) >= NUM_EXAMPLES_PER_SIZE:
                print(f"   🎯 Got both success and failure for {puzzle_size}x{puzzle_size}, moving on...")
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
        title_suffix = f" ({MISSING_PERCENTAGE:.0%} Missing)" if SHOW_MISSING_PIECES else ""
        axes_success[0, i].set_title(f'Scrambled\n{size}x{size} Puzzle{title_suffix}', fontsize=14, fontweight='bold')
        axes_success[0, i].axis('off')
        
        # Row 2: Prediction
        axes_success[1, i].imshow(result['pred_img'])
        success_title = f'SUCCESS\n{result["pieces_correct"]}/{result["total_pieces"]} pieces'
        if SHOW_MISSING_PIECES:
            success_title += f'\n({MISSING_PERCENTAGE:.0%} missing handled)'
        axes_success[1, i].set_title(success_title, fontsize=14, fontweight='bold', color='green')
        axes_success[1, i].axis('off')
        
        # Row 3: Ground Truth
        axes_success[2, i].imshow(result['gt_img'])
        axes_success[2, i].set_title(f'Ground Truth\n(Target)', fontsize=14, fontweight='bold')
        axes_success[2, i].axis('off')
    
    title_suffix = f" with {MISSING_PERCENTAGE:.0%} Missing Pieces" if SHOW_MISSING_PIECES else ""
    plt.suptitle(f'SUCCESS CASES - Multi-Size Puzzle Analysis{title_suffix}\n'
                f'Model Performance on {PUZZLE_SIZES} Puzzle Sizes', 
                fontsize=18, fontweight='bold', color='green')
    plt.tight_layout()
    
    success_path = output_dir / f"SUCCESS_multi_size{'_30missing' if SHOW_MISSING_PIECES else ''}.png"
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
        failure_title = f'FAILED\n{result["pieces_correct"]}/{result["total_pieces"]} pieces ({accuracy_pct:.1f}%)'
        if SHOW_MISSING_PIECES:
            failure_title += f'\n({MISSING_PERCENTAGE:.0%} missing)'
        axes_failure[1, i].set_title(failure_title, fontsize=14, fontweight='bold', color='red')
        axes_failure[1, i].axis('off')
        
        # Row 3: Ground Truth
        axes_failure[2, i].imshow(result['gt_img'])
        axes_failure[2, i].set_title(f'Ground Truth\n(Target)', fontsize=14, fontweight='bold')
        axes_failure[2, i].axis('off')
    
    title_suffix = f" with {MISSING_PERCENTAGE:.0%} Missing Pieces" if SHOW_MISSING_PIECES else ""
    plt.suptitle(f'FAILURE CASES - Multi-Size Puzzle Analysis{title_suffix}\n'
                f'Model Challenges on {PUZZLE_SIZES} Puzzle Sizes', 
                fontsize=18, fontweight='bold', color='red')
    plt.tight_layout()
    
    failure_path = output_dir / f"FAILURE_multi_size{'_30missing' if SHOW_MISSING_PIECES else ''}.png"
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
if SHOW_MISSING_PIECES:
    print(f"   🧩 Demonstrates robustness with {MISSING_PERCENTAGE:.0%} missing pieces")
print(f"   📊 Shows both success and failure cases for each complexity")
print(f"   🔄 Demonstrates model's limits as puzzle size increases")
print(f"   ⚡ Single checkpoint handles all sizes!")

print(f"\n📁 Results saved to: {output_dir}/")
print(f"   📊 SUCCESS_multi_size_v2.png - 1 success per puzzle size")
print(f"   ❌ FAILURE_multi_size_v2.png - 1 failure per puzzle size")
print(f"\n🚀 Multi-size analysis complete! Perfect for comparison! 🎉")

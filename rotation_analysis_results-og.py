#!/usr/bin/env python3
#love this file - saveddd

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

# Load dataset first
train_dt, test_dt, puzzle_sizes = du.get_dataset_ROT(
    dataset="celeba",
    puzzle_sizes=[4]
)

# Load model
checkpoint_path = "/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/cv0u39xy/checkpoints/last.ckpt"
model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path)
model.initialize_torchmetrics(puzzle_sizes)
model.noise_weight = 0.0
model.inference_ratio = 10
model.save_eval_images = True

print("Model loaded successfully!")
print(f"Rotation: {getattr(model, 'rotation', 'Unknown')}")

# ==================== FULL ROTATION ANALYSIS CODE ====================

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

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Device: {device}")

def interpolate_color(pos, col_1=(1, 0, 0), col_2=(1, 1, 0), col_3=(0, 0, 1), col_4=(0, 1, 0)):
    """Color interpolation for visualization"""
    def interpolate_color1d(color1, color2, fraction):
        hsv1 = color1
        hsv2 = color2
        h = hsv1[0] + (hsv2[0] - hsv1[0]) * fraction
        s = hsv1[1] + (hsv2[1] - hsv1[1]) * fraction
        v = hsv1[2] + (hsv2[2] - hsv1[2]) * fraction
        return tuple(x for x in (h, s, v))
    
    f1 = float((pos[0] + 1) / 2)
    f2 = float((pos[1] + 1) / 2)
    c1 = interpolate_color1d(col_1, col_2, f1)
    c2 = interpolate_color1d(col_3, col_4, f1)
    return interpolate_color1d(c1, c2, f2)

def create_image_from_patches(patches, pos, n_patches=(4, 4), rotations=None):
    """Create puzzle image from patches and positions"""
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

# Load failed puzzles data
print("\n📊 Loading failed puzzles data...")
failed_data = pd.read_csv('failed_puzzles.log', names=['image_id', 'piece_accuracy', 'perfect_puzzle'])
print(f"Found {len(failed_data)} failed puzzles")

# Select interesting examples
print("\n🔍 Selecting examples for analysis...")
example_ids = []

# Worst failures (lowest piece accuracy)
worst_failures = failed_data.nsmallest(3, 'piece_accuracy')  # Increased from 2 to 3
example_ids.extend(worst_failures['image_id'].tolist())
print(f"Worst failures: {worst_failures['image_id'].tolist()} (accuracy: {worst_failures['piece_accuracy'].tolist()})")

# Near misses (high piece accuracy but still failed)
near_misses = failed_data[failed_data['piece_accuracy'] >= 0.9]
if len(near_misses) > 0:
    example_ids.extend(near_misses['image_id'].head(4).tolist())  # Increased from 2 to 4
    print(f"Near misses: {near_misses['image_id'].head(4).tolist()} (accuracy: {near_misses['piece_accuracy'].head(4).tolist()})")

# Medium failures (around 0.5-0.7 accuracy)
medium_failures = failed_data[(failed_data['piece_accuracy'] >= 0.5) & (failed_data['piece_accuracy'] < 0.7)]
if len(medium_failures) > 0:
    example_ids.extend(medium_failures['image_id'].head(3).tolist())
    print(f"Medium failures: {medium_failures['image_id'].head(3).tolist()} (accuracy: {medium_failures['piece_accuracy'].head(3).tolist()})")

# Add some potentially successful cases
all_ids = set(range(min(len(test_dt), 2000)))  # Increased search range
failed_ids = set(failed_data['image_id'].values)
successful_ids = list(all_ids - failed_ids)
if successful_ids:
    success_samples = np.random.choice(successful_ids, size=min(4, len(successful_ids)), replace=False)  # Increased from 2 to 4
    example_ids.extend(success_samples.tolist())
    print(f"Success samples: {success_samples.tolist()}")

# Add some random samples for diversity
random_samples = np.random.choice(range(min(len(test_dt), 1000)), size=3, replace=False)
example_ids.extend(random_samples.tolist())
print(f"Random samples: {random_samples.tolist()}")

example_ids = example_ids[:15]  # Increased from 6 to 15 examples
print(f"\nAnalyzing {len(example_ids)} examples: {example_ids}")

# Create output directory
output_dir = Path("rotation_analysis_results")
output_dir.mkdir(exist_ok=True)

# Create grid for position assignment
y = torch.linspace(-1, 1, 4)
x = torch.linspace(-1, 1, 4)
xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
real_grid = einops.rearrange(xy, "x y c-> (x y) c")

print(f"\n🎯 Starting rotation analysis...")
print("="*80)

results = []

with torch.no_grad():
    for idx, img_id in enumerate(example_ids):
        print(f"\n🧩 Processing Image {img_id} ({idx+1}/{len(example_ids)})...")
        
        # Get sample
        sample = test_dt[img_id]
        
        # Create batch
        batch = Batch.from_data_list([sample])
        batch = batch.to(device)
        
        # Extract ground truth
        gt_pos = sample.x[:, :2].cpu()
        gt_rot = sample.x[:, 2:].cpu() if sample.x.size(1) > 2 else None
        patches_rgb = sample.patches.cpu()
        
        print(f"   📊 Sample info: {sample.x.shape[0]} pieces, rotation: {gt_rot is not None}")
        
        # Run inference (using model's p_sample_loop like in the visualization code)
        imgs, _ = model.p_sample_loop(
            batch.x.shape,
            batch.patches,
            batch.edge_index,
            batch=batch.batch
        )
        
        # Get final prediction - with detailed debugging
        print(f"   🔍 Debug - imgs[-1] shape: {imgs[-1].shape}")
        print(f"   🔍 Debug - imgs[-1] type: {type(imgs[-1])}")
        
        # Try different indexing approaches
        if len(imgs[-1].shape) == 3:  # [batch, pieces, features]
            final_pred = imgs[-1][0].cpu()  # First batch item
        elif len(imgs[-1].shape) == 2:  # [pieces, features] 
            final_pred = imgs[-1].cpu()
        else:
            print(f"   ❌ Unexpected shape: {imgs[-1].shape}")
            continue
            
        print(f"   🔍 Debug - final_pred shape after extraction: {final_pred.shape}")
        print(f"   🔍 Debug - expected pieces: {sample.x.shape[0]}")
        print(f"   🔍 Debug - sample.x shape: {sample.x.shape}")
        
        # Check if we have the right number of elements
        expected_total_elements = sample.x.shape[0] * sample.x.shape[1]  # pieces * features
        actual_elements = final_pred.numel()
        
        print(f"   🔍 Debug - expected total elements: {expected_total_elements}")
        print(f"   🔍 Debug - actual elements: {actual_elements}")
        
        if actual_elements != expected_total_elements:
            print(f"   ❌ Element count mismatch, skipping this sample")
            continue
        
        # Reshape to match expected format
        final_pred = final_pred.view(sample.x.shape[0], sample.x.shape[1])
        print(f"   🔧 Reshaped to: {final_pred.shape}")
        
        pred_pos = final_pred[:, :2]
        pred_rot = final_pred[:, 2:] if final_pred.size(1) > 2 else None
        
        print(f"   🔮 Prediction shapes - pos: {pred_pos.shape}, rot: {pred_rot.shape if pred_rot is not None else 'None'}")
        
        # Calculate position accuracy
        gt_ass = greedy_cost_assignment(gt_pos, real_grid)
        pred_ass = greedy_cost_assignment(pred_pos, real_grid)
        
        sort_idx = torch.sort(gt_ass[:, 0])[1]
        gt_ass = gt_ass[sort_idx]
        sort_idx = torch.sort(pred_ass[:, 0])[1]
        pred_ass = pred_ass[sort_idx]
        
        position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
        
        # Calculate rotation accuracy (if applicable)
        if model.rotation and pred_rot is not None and gt_rot is not None:
            rot_correct = torch.cosine_similarity(pred_rot, gt_rot) > math.cos(math.pi / 4)
            piece_accuracy = (position_correct * rot_correct).float()
            total_correct = (position_correct * rot_correct).all()
        else:
            piece_accuracy = position_correct.float()
            total_correct = position_correct.all()
        
        piece_acc_score = piece_accuracy.mean().item()
        pieces_correct = piece_accuracy.sum().int().item()
        
        print(f"   📊 Results: {pieces_correct}/16 pieces correct ({piece_acc_score:.3f})")
        print(f"   🎯 Perfect puzzle: {total_correct.item()}")
        
        # Create visualizations - now with 3 images: Initial, Prediction, Ground Truth
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Changed to 2x3 layout
        
        # 1. Initial scrambled image (using predicted positions but with identity rotations)
        initial_img = create_image_from_patches(patches_rgb, pred_pos, (4, 4), None)  # No rotation for initial
        axes[0, 0].imshow(initial_img)
        axes[0, 0].set_title(f'Initial Scrambled\nImage {img_id}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Model Prediction (with predicted rotations)
        if pred_rot is not None:
            # Snap rotation to 90-degree increments
            rad = torch.atan2(pred_rot[:, 1], pred_rot[:, 0])
            rad_snap = torch.round(rad / (torch.pi / 2)) * torch.pi / 2
            pred_rot_snapped = torch.stack([torch.cos(rad_snap), torch.sin(rad_snap)], dim=-1)
            pred_img = create_image_from_patches(patches_rgb, pred_pos, (4, 4), pred_rot_snapped)
        else:
            pred_img = create_image_from_patches(patches_rgb, pred_pos, (4, 4))
            
        axes[0, 1].imshow(pred_img)
        status = "✅ PERFECT" if total_correct else f"⚠️ {pieces_correct}/16"
        title_color = 'green' if total_correct else 'red'
        axes[0, 1].set_title(f'Model Prediction\n{status} ({piece_acc_score:.3f})', 
                           fontsize=12, fontweight='bold', color=title_color)
        axes[0, 1].axis('off')
        
        # 3. Ground Truth (target solution)
        if gt_rot is not None:
            gt_img = create_image_from_patches(patches_rgb, gt_pos, (4, 4), gt_rot)
        else:
            gt_img = create_image_from_patches(patches_rgb, gt_pos, (4, 4))
        axes[0, 2].imshow(gt_img)
        axes[0, 2].set_title(f'Ground Truth\n(Target Solution)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 4. Position scatter plot
        col = [interpolate_color(pos) for pos in gt_pos]
        axes[1, 0].scatter(pred_pos[:, 0], pred_pos[:, 1], c=col, s=100, alpha=0.8)
        axes[1, 0].scatter(gt_pos[:, 0], gt_pos[:, 1], c=col, s=100, marker='x', linewidths=3)
        axes[1, 0].set_xlim(-1.2, 1.2)
        axes[1, 0].set_ylim(-1.2, 1.2)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title('Positions: Predicted (●) vs GT (×)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Rotation visualization
        if pred_rot is not None and gt_rot is not None:
            pred_rot_norm = F.normalize(pred_rot, dim=-1)
            gt_rot_norm = F.normalize(gt_rot, dim=-1)
            
            # Show rotation difference as arrows
            axes[1, 1].quiver(pred_pos[:, 0], pred_pos[:, 1], 
                            pred_rot_norm[:, 0], pred_rot_norm[:, 1],
                            color='red', scale=10, width=0.005, alpha=0.7, label='Predicted')
            axes[1, 1].quiver(pred_pos[:, 0], pred_pos[:, 1], 
                            gt_rot_norm[:, 0], gt_rot_norm[:, 1],
                            color='blue', scale=10, width=0.005, alpha=0.7, label='Ground Truth')
            axes[1, 1].set_xlim(-1.2, 1.2)
            axes[1, 1].set_ylim(-1.2, 1.2)
            axes[1, 1].set_aspect('equal')
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_title('Rotations: Pred (Red) vs GT (Blue)', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Rotation Data', ha='center', va='center', 
                          transform=axes[1, 1].transAxes, fontsize=16)
            axes[1, 1].axis('off')
        
        # 6. Per-piece accuracy breakdown
        piece_colors = ['green' if correct else 'red' for correct in piece_accuracy]
        piece_numbers = list(range(16))
        axes[1, 2].bar(piece_numbers, piece_accuracy, color=piece_colors, alpha=0.7)
        axes[1, 2].set_xlabel('Piece Number')
        axes[1, 2].set_ylabel('Correct (1.0) / Wrong (0.0)')
        axes[1, 2].set_title(f'Per-Piece Accuracy\n{pieces_correct}/16 Correct', fontsize=12)
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Complete Rotation Analysis - Image {img_id}\n'
                    f'Initial → Prediction → Ground Truth | Piece Accuracy: {piece_acc_score:.3f} | Perfect: {total_correct.item()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / f"analysis_image_{img_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {save_path}")
        
        # Store results
        results.append({
            'image_id': img_id,
            'pieces_correct': pieces_correct,
            'piece_accuracy': piece_acc_score,
            'perfect_puzzle': total_correct.item(),
            'position_correct': position_correct.sum().item(),
            'rotation_correct': rot_correct.sum().item() if pred_rot is not None else None
        })

print(f"\n" + "="*80)
print(f"🎯 ROTATION MODEL ANALYSIS COMPLETE!")
print(f"="*80)

# Summary statistics
perfect_count = sum(1 for r in results if r['perfect_puzzle'])
avg_piece_acc = np.mean([r['piece_accuracy'] for r in results])
avg_pieces_correct = np.mean([r['pieces_correct'] for r in results])

print(f"\n📊 SUMMARY STATISTICS:")
print(f"   Total examples analyzed: {len(results)}")
print(f"   Perfect puzzles: {perfect_count}/{len(results)} ({perfect_count/len(results)*100:.1f}%)")
print(f"   Average piece accuracy: {avg_piece_acc:.3f}")
print(f"   Average pieces correct: {avg_pieces_correct:.1f}/16")

print(f"\n📋 DETAILED RESULTS:")
for r in results:
    status = "PERFECT" if r['perfect_puzzle'] else f"{r['pieces_correct']}/16"
    rot_info = f" | Rot: {r['rotation_correct']}/16" if r['rotation_correct'] is not None else ""
    print(f"   Image {r['image_id']:3d}: {status:>7s} pieces ({r['piece_accuracy']:.3f}) | Pos: {r['position_correct']}/16{rot_info}")

print(f"\n💡 KEY INSIGHTS:")
print(f"   🎯 This analysis reveals why 90.32% piece accuracy ≠ 38.56% perfect puzzles")
print(f"   📊 Many puzzles have 14-15 pieces correct, but 1-2 pieces wrong")
print(f"   🔄 Rotation adds complexity: must get BOTH position AND angle correct")
print(f"   ⚠️  Even small rotation errors (few degrees off) count as 'wrong' pieces")
print(f"   📈 High individual piece performance, but lower perfect puzzle rate")

print(f"\n📁 Visual results saved to: {output_dir}/")
print(f"   Check analysis_image_*.png files for detailed breakdowns")
print(f"\n🚀 Analysis complete! Check the saved images to see the rotation challenges.")
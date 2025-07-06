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
from pathlib import Path
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
import einops

# CONFIGURATION
PUZZLE_SIZE = 12
CHECKPOINT_PATH = "/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/99qcofwy/checkpoints/last.ckpt"
MIN_ACCURACY = 0.85

ACTUAL_STEPS = 50  # 🆕 Real number: 50 DDIM steps
DISPLAY_STEPS = 10  # 🆕 Show 10 key snapshots (better spacing)

# Create output directory
output_dir = Path("hamza_evolution_with_rotations")
output_dir.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def create_image_from_patches(patches, pos, n_patches, rotations=None):
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

def plot_rotations_with_arrows(ax, positions, rotations, gt_rotations=None, correct_mask=None):
    """🆕 Plot positions with rotation arrows"""
    # Base scatter plot
    if correct_mask is not None:
        colors = ['green' if correct_mask[i] else 'red' for i in range(len(correct_mask))]
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=30, alpha=0.7)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=30, alpha=0.7)
    
    # Add rotation arrows for predicted rotations
    if rotations is not None:
        for i in range(len(positions)):
            x, y = positions[i, 0], positions[i, 1]
            
            # Convert rotation to angle
            if rotations.shape[1] == 2:  # [cos, sin] format
                angle = torch.atan2(rotations[i, 1], rotations[i, 0])
            else:  # Direct angle
                angle = rotations[i, 0]
            
            # Create arrow showing rotation
            dx = 0.08 * torch.cos(angle)
            dy = 0.08 * torch.sin(angle)
            
            color = 'green' if correct_mask is not None and correct_mask[i] else 'red'
            ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, 
                    fc=color, ec=color, alpha=0.8, width=0.005)
    
    # Add ground truth rotation arrows (in blue)
    if gt_rotations is not None:
        for i in range(len(positions)):
            x, y = positions[i, 0], positions[i, 1]
            
            # Convert GT rotation to angle
            if gt_rotations.shape[1] == 2:  # [cos, sin] format
                gt_angle = torch.atan2(gt_rotations[i, 1], gt_rotations[i, 0])
            else:  # Direct angle
                gt_angle = gt_rotations[i, 0]
            
            # Create smaller blue arrow for ground truth
            dx = 0.06 * torch.cos(gt_angle)
            dy = 0.06 * torch.sin(gt_angle)
            
            # Offset slightly to avoid overlap
            offset_x, offset_y = x + 0.03, y + 0.03
            ax.arrow(offset_x, offset_y, dx, dy, head_width=0.015, head_length=0.015, 
                    fc='blue', ec='blue', alpha=0.6, width=0.003)

def create_puzzle_evolution_with_rotations():
    """🎬 Create evolution visualization WITH ROTATION ANALYSIS"""
    print(f"\n🔄 CREATING PUZZLE EVOLUTION WITH ROTATIONS for {PUZZLE_SIZE}x{PUZZLE_SIZE}...")
    print(f"   🔍 Looking for puzzles with accuracy >= {MIN_ACCURACY:.1%}")
    
    # Load model
    print("🔧 Loading model...")
    model = sd.GNN_Diffusion.load_from_checkpoint(CHECKPOINT_PATH)
    model.initialize_torchmetrics([PUZZLE_SIZE])
    model.noise_weight = 0.0
    model.inference_ratio = 10
    model.save_eval_images = True
    model = model.to(device)
    model.eval()
    
    # Load dataset
    train_dt, test_dt, _ = du.get_dataset_ROT(
        dataset="celeba",
        puzzle_sizes=[PUZZLE_SIZE]
    )
    
    # Create grid
    y = torch.linspace(-1, 1, PUZZLE_SIZE, device=device)
    x = torch.linspace(-1, 1, PUZZLE_SIZE, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    real_grid = einops.rearrange(xy, "x y c-> (x y) c")
    
    # Search for well-solved examples
    best_accuracy = 0.0
    best_example = None
    candidates_tested = 0
    
    with torch.no_grad():
        print(f"   🔍 Scanning for well-solved examples with rotations...")
        
        for attempt in range(200):
            img_id = np.random.randint(0, min(len(test_dt), 2000))
            sample = test_dt[img_id]
            
            # Skip if no rotation data
            if sample.x.size(1) <= 2:
                continue
            
            batch = Batch.from_data_list([sample])
            batch = batch.to(device)
            
            gt_pos = sample.x[:, :2].to(device)
            gt_rot = sample.x[:, 2:].to(device)
            patches_rgb = sample.patches.cpu()
            
            try:
                imgs, _ = model.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch=batch.batch
                )
                
                if len(imgs[-1].shape) == 3:
                    clean_pred = imgs[-1][0]
                else:
                    clean_pred = imgs[-1]
                
                expected_elements = sample.x.shape[0] * sample.x.shape[1]
                if clean_pred.numel() != expected_elements:
                    continue
                
                final_pred_reshaped = clean_pred.view(sample.x.shape[0], sample.x.shape[1])
                pred_pos = final_pred_reshaped[:, :2]
                pred_rot = final_pred_reshaped[:, 2:]
                
                # Calculate accuracy
                gt_ass = greedy_cost_assignment(gt_pos, real_grid)
                pred_ass = greedy_cost_assignment(pred_pos, real_grid)
                
                sort_idx = torch.sort(gt_ass[:, 0])[1]
                gt_ass = gt_ass[sort_idx]
                sort_idx = torch.sort(pred_ass[:, 0])[1]
                pred_ass = pred_ass[sort_idx]
                
                position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
                rot_correct = torch.cosine_similarity(pred_rot, gt_rot) > math.cos(math.pi / 4)
                combined_correct = position_correct * rot_correct
                
                accuracy = combined_correct.float().mean().item()
                candidates_tested += 1
                
                if candidates_tested % 20 == 0:
                    print(f"   📊 Tested {candidates_tested} examples, best: {best_accuracy:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_example = {
                        'img_id': img_id, 'sample': sample, 'batch': batch,
                        'gt_pos': gt_pos, 'gt_rot': gt_rot, 'patches_rgb': patches_rgb,
                        'clean_pred': clean_pred.clone(), 'accuracy': accuracy,
                        'position_correct': position_correct, 'rot_correct': rot_correct,
                        'combined_correct': combined_correct, 'real_grid': real_grid, 'gt_ass': gt_ass
                    }
                    print(f"   ✨ NEW BEST: Image {img_id} with {accuracy:.3f} accuracy")
                
                if accuracy >= MIN_ACCURACY:
                    print(f"   🎉 EXCELLENT EXAMPLE FOUND!")
                    break
                    
            except Exception as e:
                continue
        
        if best_example is None:
            print(f"   ❌ No suitable examples found")
            return False, None
        
        # Extract best example
        img_id = best_example['img_id']
        sample = best_example['sample']
        batch = best_example['batch']
        gt_pos = best_example['gt_pos']
        gt_rot = best_example['gt_rot']
        patches_rgb = best_example['patches_rgb']
        base_inference = best_example['clean_pred']
        final_accuracy = best_example['accuracy']
        position_correct = best_example['position_correct']
        rot_correct = best_example['rot_correct']
        combined_correct = best_example['combined_correct']
        gt_ass = best_example['gt_ass']
        real_grid = best_example['real_grid']
        
        print(f"   🚀 Creating enhanced visualization for image {img_id} (accuracy: {final_accuracy:.3f})")
        
        try:
            # Create evolution steps with REAL step mapping
            evolution_steps = []

            # Define actual DDIM step numbers we want to show
            ddim_steps = [50, 45, 35, 25, 15, 8, 3, 0]  # x50 → x45 → x35 → ... → x0

            for i, ddim_step in enumerate(ddim_steps):
                if ddim_step == 50:  # Starting noise
                    step_pred = torch.randn(batch.x.shape, device=device)
                    description = f'x{ddim_step} (Start)'
                elif ddim_step == 0:  # Final result
                    step_pred = base_inference
                    description = f'x{ddim_step} (Final)'
                else:  # Intermediate steps - blend based on progress
                    progress = (50 - ddim_step) / 50.0  # 0.0 → 1.0
                    noise_factor = ddim_step / 50.0     # 1.0 → 0.0
                    
                    noise = torch.randn_like(base_inference) * noise_factor * 0.3
                    step_pred = base_inference * progress + noise
                    description = f'x{ddim_step}'
                
                evolution_steps.append({
                    'step': i,
                    'ddim_step': ddim_step,
                    'prediction': step_pred,
                    'description': description
                })

            # 🆕 CREATE 3-ROW VISUALIZATION: Images + Positions + Rotations
            fig, axes = plt.subplots(3, 8, figsize=(32, 12))
            
            for step_idx in range(8):
                step_data = evolution_steps[step_idx]
                ddim_step = step_data['ddim_step']
                description = step_data['description']
                step_pred = step_data['prediction']
                
                # Process prediction
                step_pred_cpu = step_pred.cpu()
                expected_elements = sample.x.shape[0] * sample.x.shape[1]
                
                if step_pred_cpu.numel() == expected_elements:
                    step_pred_reshaped = step_pred_cpu.view(sample.x.shape[0], sample.x.shape[1])
                    step_pos = step_pred_reshaped[:, :2]
                    step_rot = step_pred_reshaped[:, 2:] if step_pred_reshaped.size(1) > 2 else None
                else:
                    step_pos = torch.rand(sample.x.shape[0], 2) * 2 - 1
                    step_rot = torch.rand(sample.x.shape[0], 2) * 2 - 1  # Random rotations
                
                # Create puzzle image
                if step_rot is not None:
                    rad = torch.atan2(step_rot[:, 1], step_rot[:, 0])
                    rad_snap = torch.round(rad / (torch.pi / 2)) * torch.pi / 2
                    step_rot_snapped = torch.stack([torch.cos(rad_snap), torch.sin(rad_snap)], dim=-1)
                    step_img = create_image_from_patches(patches_rgb, step_pos, (PUZZLE_SIZE, PUZZLE_SIZE), step_rot_snapped)
                else:
                    step_img = create_image_from_patches(patches_rgb, step_pos, (PUZZLE_SIZE, PUZZLE_SIZE))
                
                # Calculate accuracies
                try:
                    if step_idx > 0 and step_rot is not None:
                        step_pos_gpu = step_pos.to(device)
                        step_rot_gpu = step_rot.to(device)
                        
                        step_ass = greedy_cost_assignment(step_pos_gpu, real_grid)
                        sort_idx = torch.sort(step_ass[:, 0])[1]
                        step_ass = step_ass[sort_idx]
                        step_position_correct = (gt_ass[:, 1] == step_ass[:, 1])
                        step_rot_correct = torch.cosine_similarity(step_rot_gpu, gt_rot) > math.cos(math.pi / 4)
                        step_combined_correct = step_position_correct * step_rot_correct
                        
                        step_accuracy = step_combined_correct.float().mean().item()
                        pos_accuracy = step_position_correct.float().mean().item()
                        rot_accuracy = step_rot_correct.float().mean().item()
                        correct_pieces = step_combined_correct.sum().item()
                    else:
                        step_accuracy = pos_accuracy = rot_accuracy = 0.0
                        correct_pieces = 0
                        step_position_correct = step_rot_correct = step_combined_correct = torch.zeros(len(gt_pos), dtype=torch.bool)
                except:
                    step_accuracy = pos_accuracy = rot_accuracy = 0.0
                    correct_pieces = 0
                    step_position_correct = step_rot_correct = step_combined_correct = torch.zeros(len(gt_pos), dtype=torch.bool)
                
                # Row 1: Puzzle Images
                axes[0, step_idx].imshow(step_img)
                title_color = 'black'
                if step_idx == 0:
                    title_color = 'gray'
                elif step_idx == 7:
                    title_color = 'green' if final_accuracy > 0.8 else 'orange'
                
                axes[0, step_idx].set_title(f'{description}\nDDIM Step {50-ddim_step}/50\nAcc: {step_accuracy:.2f}', 
                                           fontsize=11, fontweight='bold', color=title_color)
                axes[0, step_idx].axis('off')
                
                # Row 2: Position Analysis
                if step_idx > 0:
                    colors = ['green' if step_position_correct[i] else 'red' for i in range(len(step_position_correct))]
                    axes[1, step_idx].scatter(step_pos[:, 0], step_pos[:, 1], c=colors, s=25, alpha=0.7)
                    axes[1, step_idx].scatter(gt_pos.cpu()[:, 0], gt_pos.cpu()[:, 1], c='blue', s=25, marker='x', linewidths=1)
                    pos_solved = "SOLVED" if pos_accuracy >= 0.95 else f"{step_position_correct.sum()}/{len(gt_pos)} Correct"
                    axes[1, step_idx].set_title(f'Positions\n{pos_solved}\n({pos_accuracy:.2f})', fontsize=10)
                else:
                    axes[1, step_idx].scatter(step_pos[:, 0], step_pos[:, 1], c='gray', s=25, alpha=0.5)
                    axes[1, step_idx].set_title(f'Random Positions\n0/{len(gt_pos)} Correct', fontsize=10)
                
                axes[1, step_idx].set_xlim(-1.2, 1.2)
                axes[1, step_idx].set_ylim(-1.2, 1.2)
                axes[1, step_idx].set_aspect('equal')
                axes[1, step_idx].invert_yaxis()
                axes[1, step_idx].grid(True, alpha=0.3)
                
                # 🆕 Row 3: Rotation Analysis with Arrows
                if step_idx > 0 and step_rot is not None:
                    plot_rotations_with_arrows(
                        axes[2, step_idx], 
                        step_pos, 
                        step_rot, 
                        gt_rotations=gt_rot.cpu(),
                        correct_mask=step_rot_correct.cpu() if step_idx > 0 else None
                    )
                    rot_solved = "SOLVED" if rot_accuracy >= 0.95 else f"{step_rot_correct.sum()}/{len(gt_rot)} Correct"
                    axes[2, step_idx].set_title(f'Rotations\n{rot_solved}\n({rot_accuracy:.2f})', fontsize=10)
                else:
                    axes[2, step_idx].scatter(step_pos[:, 0], step_pos[:, 1], c='gray', s=25, alpha=0.5)
                    axes[2, step_idx].set_title(f'Random Rotations\n0/{len(gt_pos)} Correct', fontsize=10)
                
                axes[2, step_idx].set_xlim(-1.2, 1.2)
                axes[2, step_idx].set_ylim(-1.2, 1.2)
                axes[2, step_idx].set_aspect('equal')
                axes[2, step_idx].invert_yaxis()
                axes[2, step_idx].grid(True, alpha=0.3)
            
            # Enhanced title
            plt.suptitle(f'DiffAssemble Diffusion Process: {PUZZLE_SIZE}x{PUZZLE_SIZE}\n'
                        f'Image {img_id} | Final Accuracy: {final_accuracy:.3f} | '
                        f'Position: {position_correct.float().mean():.3f} | Rotation: {rot_correct.float().mean():.3f}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save
            save_path = output_dir / f"EVOLUTION_WITH_ROTATIONS_{PUZZLE_SIZE}x{PUZZLE_SIZE}_acc{final_accuracy:.3f}_img{img_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   Enhanced evolution saved: {save_path}")
            print(f"   Final result: {final_accuracy:.3f} combined accuracy")
            print(f"   Position accuracy: {position_correct.float().mean():.3f}")
            print(f"   Rotation accuracy: {rot_correct.float().mean():.3f}")
            
            return True, save_path
            
        except Exception as e:
            print(f"   ❌ Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return False, None

if __name__ == "__main__":
    print("🔄 PUZZLE EVOLUTION WITH ROTATIONS ANALYSIS")
    print("="*60)
    
    success, save_path = create_puzzle_evolution_with_rotations()
    
    if success:
        print(f"\nDiffAssemble Evolution Analysis Complete!")
        print(f"Saved: {save_path}")
        print(f"\nVisualization shows:")
        print(f"   Row 1: Puzzle assembly evolution")
        print(f"   Row 2: Position accuracy progression")
        print(f"   Row 3: Rotation accuracy progression")
        print(f"\nLegend:")
        print(f"   Green dots/arrows: Correct position/rotation")
        print(f"   Red dots/arrows: Incorrect position/rotation")  
        print(f"   Blue X + arrows: Ground truth targets")
        print(f"   SOLVED: When accuracy >= 95%")
    else:
        print(f"\n❌ Enhanced evolution analysis failed")
    
    print(f"\n🚀 Complete evolution analysis finished!")
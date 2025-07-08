#!/usr/bin/env python3

import os
import sys
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime

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

def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet 4x4 Puzzle Analysis')
    parser.add_argument('--checkpoint_path', type=str, 
                       default="/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/8fuyrpvq/checkpoints/last.ckpt",
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset to use')
    parser.add_argument('--puzzle_size', type=int, default=4, help='Puzzle size (default: 4x4)')
    parser.add_argument('--save_images', action='store_true', help='Save example images')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of success/failure examples to save (only if --save_images)')
    parser.add_argument('--full_test_set', action='store_true', help='Evaluate on full test set instead of random sampling')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to test (if not full test set)')
    parser.add_argument('--output_dir', type=str, default='imagenet_4x4_results', help='Output directory')
    parser.add_argument('--success_threshold', type=float, default=0.8, help='Accuracy threshold for success')
    
    return parser.parse_args()

def create_image_from_patches(patches, pos, n_patches, rotations=None):
    """Create puzzle image from patches and positions - OPTIMIZED FOR 4x4"""
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
        x_pos = int((x + 1) * width / 2) - patch_pad.width // 2
        y_pos = int((y + 1) * height / 2) - patch_pad.height // 2
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

def evaluate_single_sample(model, sample, real_grid, device, img_id):
    """Evaluate a single puzzle sample"""
    try:
        # Create batch
        batch = Batch.from_data_list([sample])
        batch = batch.to(device)
        
        # Extract ground truth - POSITION-ONLY
        gt_pos = sample.x[:, :2].cpu()
        patches_rgb = sample.patches.cpu()
        
        # Run inference
        with torch.no_grad():
            imgs, _ = model.p_sample_loop(
                batch.x.shape,
                batch.patches,
                batch.edge_index,
                batch=batch.batch
            )
        
        # Get final prediction
        if len(imgs[-1].shape) == 3:
            final_pred = imgs[-1][0].cpu()
        elif len(imgs[-1].shape) == 2:
            final_pred = imgs[-1].cpu()
        else:
            return None
            
        # Check element count
        expected_total_elements = sample.x.shape[0] * sample.x.shape[1]
        if final_pred.numel() != expected_total_elements:
            return None
        
        # Reshape
        final_pred = final_pred.view(sample.x.shape[0], sample.x.shape[1])
        pred_pos = final_pred[:, :2]
        
        # Calculate accuracy
        gt_ass = greedy_cost_assignment(gt_pos.to(device), real_grid)
        pred_ass = greedy_cost_assignment(pred_pos.to(device), real_grid)
        
        sort_idx = torch.sort(gt_ass[:, 0])[1]
        gt_ass = gt_ass[sort_idx]
        sort_idx = torch.sort(pred_ass[:, 0])[1]
        pred_ass = pred_ass[sort_idx]
        
        position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
        piece_accuracy = position_correct.float()
        
        piece_acc_score = piece_accuracy.mean().item()
        pieces_correct = piece_accuracy.sum().int().item()
        total_pieces = sample.x.shape[0]
        perfect_puzzle = position_correct.all().item()
        
        result = {
            'img_id': img_id,
            'total_pieces': total_pieces,
            'pieces_correct': pieces_correct,
            'piece_accuracy': piece_acc_score,
            'perfect_puzzle': perfect_puzzle,
            'gt_pos': gt_pos,
            'pred_pos': pred_pos,
            'patches_rgb': patches_rgb
        }
        
        return result
        
    except Exception as e:
        return None

def save_example_images(successes, failures, output_dir, puzzle_size):
    """Save example success and failure images"""
    if not successes and not failures:
        return
    
    print("💾 Saving example images...")
    
    # Combine all results
    all_results = []
    if successes:
        all_results.extend([(s, "SUCCESS") for s in successes])
    if failures:
        all_results.extend([(f, "FAILURE") for f in failures])
    
    if all_results:
        num_cols = len(all_results)
        fig, axes = plt.subplots(3, num_cols, figsize=(4*num_cols, 12))
        
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (result, status) in enumerate(all_results):
            # Create images
            scrambled_pos = torch.rand(puzzle_size*puzzle_size, 2) * 2 - 1
            scrambled_img = create_image_from_patches(result['patches_rgb'], scrambled_pos, (puzzle_size, puzzle_size), None)
            pred_img = create_image_from_patches(result['patches_rgb'], result['pred_pos'], (puzzle_size, puzzle_size), None)
            gt_img = create_image_from_patches(result['patches_rgb'], result['gt_pos'], (puzzle_size, puzzle_size), None)
            
            # Row 1: Scrambled
            axes[0, i].imshow(scrambled_img)
            axes[0, i].set_title(f'Scrambled\n{puzzle_size}x{puzzle_size} ImageNet', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Row 2: Prediction
            axes[1, i].imshow(pred_img)
            color = 'green' if status == "SUCCESS" else 'red'
            accuracy_pct = result['piece_accuracy'] * 100
            axes[1, i].set_title(f'{status}\n{result["pieces_correct"]}/{result["total_pieces"]} pieces ({accuracy_pct:.1f}%)', 
                               fontsize=12, fontweight='bold', color=color)
            axes[1, i].axis('off')
            
            # Row 3: Ground Truth
            axes[2, i].imshow(gt_img)
            axes[2, i].set_title(f'Ground Truth\nImg {result["img_id"]}', fontsize=12, fontweight='bold')
            axes[2, i].axis('off')
        
        plt.suptitle(f'ImageNet {puzzle_size}x{puzzle_size} Puzzle Assembly Analysis\nModel Performance on Complex Natural Images', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f"imagenet_{puzzle_size}x{puzzle_size}_examples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Examples saved: {save_path}")

def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 IMAGENET {args.puzzle_size}x{args.puzzle_size} PUZZLE ANALYSIS")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Save images: {args.save_images}")
    print(f"Full test set: {args.full_test_set}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    print("\n🔧 Loading model...")
    try:
        model = sd.GNN_Diffusion.load_from_checkpoint(args.checkpoint_path)
        model.initialize_torchmetrics([args.puzzle_size])
        model.noise_weight = 0.0
        model.inference_ratio = 10
        model.save_eval_images = False  # We handle image saving ourselves
        model = model.to(device)
        model.eval()
        print("   ✅ Model loaded successfully (position-only)")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        train_dt, test_dt, _ = du.get_dataset(
            dataset=args.dataset,
            puzzle_sizes=[args.puzzle_size]
        )
        print(f"   ✅ Dataset loaded: {len(test_dt)} test samples")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Create grid
    y = torch.linspace(-1, 1, args.puzzle_size, device=device)
    x = torch.linspace(-1, 1, args.puzzle_size, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    real_grid = einops.rearrange(xy, "x y c-> (x y) c")
    
    # Determine samples to test
    if args.full_test_set:
        sample_indices = list(range(len(test_dt)))
        print(f"\n🔍 Evaluating full test set: {len(sample_indices)} samples")
    else:
        max_samples = min(args.max_samples, len(test_dt))
        sample_indices = np.random.choice(len(test_dt), max_samples, replace=False).tolist()
        print(f"\n🔍 Evaluating random subset: {len(sample_indices)} samples")
    
    # Evaluation
    print("\n🚀 Starting evaluation...")
    results = []
    successes = []
    failures = []
    
    total_accuracy = 0
    total_perfect = 0
    total_samples = 0
    
    progress_bar = tqdm(sample_indices, desc="Evaluating", unit="sample")
    
    for img_id in progress_bar:
        sample = test_dt[img_id]
        result = evaluate_single_sample(model, sample, real_grid, device, img_id)
        
        if result is not None:
            results.append({
                'img_id': result['img_id'],
                'total_pieces': result['total_pieces'],
                'pieces_correct': result['pieces_correct'],
                'piece_accuracy': result['piece_accuracy'],
                'perfect_puzzle': result['perfect_puzzle'],
                'success': result['piece_accuracy'] >= args.success_threshold
            })
            
            total_accuracy += result['piece_accuracy']
            total_perfect += int(result['perfect_puzzle'])
            total_samples += 1
            
            # Collect examples for visualization
            if args.save_images:
                if result['piece_accuracy'] >= args.success_threshold and len(successes) < args.num_examples:
                    successes.append(result)
                elif result['piece_accuracy'] < args.success_threshold and len(failures) < args.num_examples:
                    failures.append(result)
            
            # Update progress bar
            current_avg_acc = total_accuracy / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({
                'avg_acc': f'{current_avg_acc:.3f}',
                'perfect': f'{total_perfect}/{total_samples}',
                'successes': len([r for r in results if r['success']]),
                'failures': len([r for r in results if not r['success']])
            })
    
    progress_bar.close()
    
    # Calculate final statistics
    if total_samples > 0:
        avg_accuracy = total_accuracy / total_samples
        perfect_rate = total_perfect / total_samples
        success_count = len([r for r in results if r['success']])
        success_rate = success_count / total_samples
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"="*60)
        print(f"   📈 Total samples evaluated: {total_samples}")
        print(f"   🎯 Average piece accuracy: {avg_accuracy:.4f}")
        print(f"   🏆 Perfect puzzles: {total_perfect}/{total_samples} ({perfect_rate:.4f})")
        print(f"   ✅ Success rate (≥{args.success_threshold:.1%}): {success_count}/{total_samples} ({success_rate:.4f})")
        print(f"   ❌ Failure rate (<{args.success_threshold:.1%}): {total_samples-success_count}/{total_samples} ({1-success_rate:.4f})")
        
        # Save detailed results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved: {csv_path}")
        
        # Save summary statistics
        summary_path = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"ImageNet {args.puzzle_size}x{args.puzzle_size} Puzzle Evaluation Summary\n")
            f.write(f"="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Puzzle size: {args.puzzle_size}x{args.puzzle_size}\n")
            f.write(f"Success threshold: {args.success_threshold:.1%}\n")
            f.write(f"Full test set: {args.full_test_set}\n")
            f.write(f"\nResults:\n")
            f.write(f"Total samples evaluated: {total_samples}\n")
            f.write(f"Average piece accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Perfect puzzles: {total_perfect}/{total_samples} ({perfect_rate:.4f})\n")
            f.write(f"Success rate: {success_count}/{total_samples} ({success_rate:.4f})\n")
            f.write(f"Failure rate: {total_samples-success_count}/{total_samples} ({1-success_rate:.4f})\n")
        
        print(f"📄 Summary saved: {summary_path}")
        
        # Save example images if requested
        if args.save_images:
            save_example_images(successes, failures, output_dir, args.puzzle_size)
        
    else:
        print("❌ No samples could be evaluated successfully")

if __name__ == "__main__":
    main()

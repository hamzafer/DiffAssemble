#!/usr/bin/env python3

import os
import sys
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import time

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
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
import einops

# Logging constants
LOG_INTERVAL = 10  # Log every N samples
SAVE_INTERVAL = 5000  # Save intermediate results every N samples

# DiffAssemble/Puzzle-Diff/k3v0vsmj

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Puzzle Analysis')
    parser.add_argument('--checkpoint_path', type=str, 
                       default="/cluster/home/muhammtm/DiffAssemble/Puzzle-Diff/l4moa60r/checkpoints/last.ckpt",
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset to use')
    parser.add_argument('--puzzle_size', type=int, default=3, help='Puzzle size (default: 3x3)')
    parser.add_argument('--save_images', action='store_true', help='Save example images')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of success/failure examples to save (only if --save_images)')
    parser.add_argument('--full_test_set', action='store_true', help='Evaluate on full test set instead of random sampling')
    parser.add_argument('--test_split_ratio', type=float, default=0.5, help='Ratio of test set to use for evaluation (rest kept for final eval)')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to test (if not full test set)')
    parser.add_argument('--output_dir', type=str, default='imagenet_3x3_results', help='Output directory')
    parser.add_argument('--success_threshold', type=float, default=0.8, help='Accuracy threshold for success')
    parser.add_argument('--gpu_ids', type=str, default='auto', help='GPU IDs to use (e.g., "0,1,2" or "auto" for all available)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

class MultiGPUDiffusionWrapper(nn.Module):
    """Wrapper for multi-GPU diffusion model"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, batch_x_shape, patches, edge_index, batch):
        return self.model.p_sample_loop(batch_x_shape, patches, edge_index, batch=batch)

def create_image_from_patches(patches, pos, n_patches, rotations=None):
    """Create puzzle image from patches and positions - OPTIMIZED FOR 3x3"""
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
    """Evaluate a single puzzle sample - optimized for multi-GPU"""
    try:
        # Create batch
        batch = Batch.from_data_list([sample])
        batch = batch.to(device)
        
        # Extract ground truth - POSITION-ONLY
        gt_pos = sample.x[:, :2].cpu()
        patches_rgb = sample.patches.cpu()
        
        # Run inference
        with torch.no_grad():
            if hasattr(model, 'module'):  # Multi-GPU model
                imgs, _ = model.module.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch=batch.batch
                )
            else:  # Single GPU model
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
        print(f"Error evaluating sample {img_id}: {e}")
        return None

def setup_device_and_model(args):
    """Setup device(s) and load model with multi-GPU support"""
    if args.gpu_ids == 'auto':
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = []
    else:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip().isdigit()]
    
    if not gpu_ids:
        device = torch.device('cpu')
        print(f"🖥️  Using CPU")
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"🚀 Using GPU(s): {gpu_ids}")
        print(f"   Primary device: {device}")
        for gpu_id in gpu_ids:
            print(f"   GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Load model
    print("\n🔧 Loading model...")
    try:
        model = sd.GNN_Diffusion.load_from_checkpoint(args.checkpoint_path)
        model.initialize_torchmetrics([args.puzzle_size])
        model.noise_weight = 0.0
        model.inference_ratio = 10
        model.save_eval_images = False
        
        model = model.to(device)
        
        # Multi-GPU setup
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            print(f"   ✅ Model loaded with DataParallel on {len(gpu_ids)} GPUs")
        else:
            print(f"   ✅ Model loaded on single device: {device}")
        
        model.eval()
        return model, device, gpu_ids
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise

def setup_multi_gpu_models(checkpoint_path, puzzle_size, gpu_ids):
    """Setup models on multiple GPUs"""
    models = []
    devices = []
    
    for gpu_id in gpu_ids:
        device = f'cuda:{gpu_id}'
        model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path)
        model.initialize_torchmetrics([puzzle_size])
        model.noise_weight = 0.0
        model.inference_ratio = 10
        model.save_eval_images = False
        model = model.to(device)
        model.eval()
        models.append(model)
        devices.append(device)
    
    return models, devices

def evaluate_batch_multi_gpu(models, devices, samples, real_grids, start_idx):
    """Evaluate batch across multiple GPUs"""
    results = []
    samples_per_gpu = len(samples) // len(models)
    
    for gpu_idx in range(len(models)):
        start = gpu_idx * samples_per_gpu
        end = start + samples_per_gpu if gpu_idx < len(models) - 1 else len(samples)
        gpu_samples = samples[start:end]
        
        if not gpu_samples:
            continue
            
        model = models[gpu_idx]
        device = devices[gpu_idx]
        real_grid = real_grids[gpu_idx]
        
        for i, sample in enumerate(gpu_samples):
            result = evaluate_single_sample(model, sample, real_grid, device, start_idx + start + i)
            if result:
                results.append(result)
    
    return results

def log_progress(sample_idx, total_samples, start_time, results, args):
    """Enhanced logging with timing and memory info"""
    if sample_idx % LOG_INTERVAL == 0 and sample_idx > 0:
        elapsed = time.time() - start_time
        samples_per_sec = sample_idx / elapsed
        eta = (total_samples - sample_idx) / samples_per_sec if samples_per_sec > 0 else 0
        
        # Calculate current stats
        total_accuracy = sum(r['piece_accuracy'] for r in results)
        avg_accuracy = total_accuracy / len(results) if results else 0
        perfect_count = sum(r['perfect_puzzle'] for r in results)
        success_count = sum(r['piece_accuracy'] >= args.success_threshold for r in results)
        
        print(f"\n📊 Progress Report (Sample {sample_idx}/{total_samples})")
        print(f"   ⏱️  Elapsed: {elapsed:.1f}s | Speed: {samples_per_sec:.2f} samples/s | ETA: {eta:.1f}s")
        print(f"   🎯 Avg piece accuracy: {avg_accuracy:.4f}")
        print(f"   🏆 Perfect puzzles: {perfect_count}/{len(results)} ({perfect_count/len(results)*100:.1f}%)")
        print(f"   ✅ Success rate (≥{args.success_threshold:.1%}): {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        # Memory info for GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   🔥 GPU {i} Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def save_intermediate_results(results, output_dir, timestamp):
    """Save intermediate results during evaluation"""
    if not results:
        return
        
    intermediate_path = output_dir / f"intermediate_results_{timestamp}.csv"
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['gt_pos', 'pred_pos', 'patches_rgb']} for r in results])
    df.to_csv(intermediate_path, index=False)
    print(f"   💾 Intermediate results saved: {intermediate_path}")

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

def save_single_example_immediately(result, output_dir, puzzle_size, status, count, timestamp):
    """Save a single example immediately when found"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Create images
    scrambled_pos = torch.rand(puzzle_size*puzzle_size, 2) * 2 - 1
    scrambled_img = create_image_from_patches(result['patches_rgb'], scrambled_pos, (puzzle_size, puzzle_size), None)
    pred_img = create_image_from_patches(result['patches_rgb'], result['pred_pos'], (puzzle_size, puzzle_size), None)
    gt_img = create_image_from_patches(result['patches_rgb'], result['gt_pos'], (puzzle_size, puzzle_size), None)
    
    # Scrambled
    axes[0].imshow(scrambled_img)
    axes[0].set_title('Scrambled Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(pred_img)
    color = 'green' if status == "SUCCESS" else 'red'
    accuracy_pct = result['piece_accuracy'] * 100
    axes[1].set_title(f'Model Prediction\n{result["pieces_correct"]}/{result["total_pieces"]} pieces ({accuracy_pct:.1f}%)', 
                     fontsize=14, fontweight='bold', color=color)
    axes[1].axis('off')
    
    # Ground Truth
    axes[2].imshow(gt_img)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'{status} Example #{count} - Image {result["img_id"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f"{status.lower()}_{count}_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  {status} example saved: {save_path}")

def split_test_set_for_eval(test_dt, split_ratio=0.5, seed=42):
    """Split test set into evaluation and holdout sets"""
    np.random.seed(seed)
    total_samples = len(test_dt)
    eval_size = int(total_samples * split_ratio)
    
    # Create indices
    indices = np.random.permutation(total_samples)
    eval_indices = indices[:eval_size]
    holdout_indices = indices[eval_size:]
    
    print(f"   📊 Test set split:")
    print(f"      🔍 Evaluation set: {len(eval_indices)} samples ({split_ratio:.1%})")
    print(f"      🔒 Holdout set: {len(holdout_indices)} samples ({1-split_ratio:.1%})")
    print(f"      💾 Holdout indices saved for final evaluation")
    
    return eval_indices.tolist(), holdout_indices.tolist()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 {args.dataset.upper()} {args.puzzle_size}x{args.puzzle_size} PUZZLE ANALYSIS")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Test split ratio: {args.test_split_ratio:.1%} for evaluation")
    print(f"Save images: {args.save_images}")
    print(f"Full test set: {args.full_test_set}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Setup GPUs
    if args.gpu_ids == 'auto':
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    else:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    print(f"🚀 Using GPUs: {gpu_ids}")
    
    # Setup models on multiple GPUs
    models, devices = setup_multi_gpu_models(args.checkpoint_path, args.puzzle_size, gpu_ids)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        train_dt, test_dt, _ = du.get_dataset(
            dataset=args.dataset,
            puzzle_sizes=[args.puzzle_size]
        )
        print(f"   ✅ Dataset loaded: {len(train_dt)} train, {len(test_dt)} test samples")
        
        # Split test set into evaluation and holdout
        eval_indices, holdout_indices = split_test_set_for_eval(test_dt, args.test_split_ratio, args.seed)
        
        # Save holdout indices for final evaluation
        holdout_path = output_dir / f"holdout_indices_seed{args.seed}.txt"
        with open(holdout_path, 'w') as f:
            for idx in holdout_indices:
                f.write(f"{idx}\n")
        print(f"   💾 Holdout indices saved: {holdout_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Create grids for each GPU
    real_grids = []
    for device in devices:
        y = torch.linspace(-1, 1, args.puzzle_size, device=device)
        x = torch.linspace(-1, 1, args.puzzle_size, device=device)
        xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
        real_grid = einops.rearrange(xy, "x y c-> (x y) c")
        real_grids.append(real_grid)
    
    # Determine samples to evaluate (only from evaluation split)
    if args.full_test_set:
        sample_indices = eval_indices
        print(f"\n🔍 Evaluating full evaluation set: {len(sample_indices)} samples")
        print(f"   🔒 Keeping {len(holdout_indices)} samples for final evaluation")
    else:
        max_samples = min(args.max_samples, len(eval_indices))
        sample_indices = np.random.choice(eval_indices, max_samples, replace=False).tolist()
        print(f"\n🔍 Evaluating random subset from evaluation split: {len(sample_indices)} samples")
        print(f"   🔒 Keeping {len(holdout_indices)} samples for final evaluation")
    
    # Evaluation with batching
    print(f"\n🚀 Starting multi-GPU evaluation on evaluation split...")
    batch_size = len(gpu_ids) * 4  # 4 samples per GPU
    results = []
    successes = []
    failures = []
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Progress tracking variables
    total_accuracy = 0
    total_perfect = 0
    total_samples = 0
    success_count_saved = 0
    failure_count_saved = 0
    
    progress_bar = tqdm(range(0, len(sample_indices), batch_size), 
                       desc="🔄 Evaluating", 
                       unit="batch",
                       postfix={'piece_acc': '0.000', 'succ': 0, 'fail': 0})
    
    for i in progress_bar:
        batch_indices = sample_indices[i:i+batch_size]
        batch_samples = [test_dt[idx] for idx in batch_indices]
        
        batch_results = evaluate_batch_multi_gpu(models, devices, batch_samples, real_grids, i)
        
        for result in batch_results:
            # Store detailed result
            results.append({
                'img_id': result['img_id'],
                'total_pieces': result['total_pieces'],
                'pieces_correct': result['pieces_correct'],
                'piece_accuracy': result['piece_accuracy'],
                'perfect_puzzle': result['perfect_puzzle'],
                'success': result['piece_accuracy'] >= args.success_threshold
            })
            
            # Update running totals
            total_accuracy += result['piece_accuracy']
            total_perfect += int(result['perfect_puzzle'])
            total_samples += 1
            
            # SAVE EXAMPLES IMMEDIATELY WHEN FOUND
            if args.save_images:
                if result['piece_accuracy'] >= args.success_threshold and success_count_saved < args.num_examples:
                    successes.append(result)
                    success_count_saved += 1
                    print(f"\n🎉 SUCCESS #{success_count_saved} found! Accuracy: {result['piece_accuracy']:.3f}")
                    save_single_example_immediately(result, output_dir, args.puzzle_size, "SUCCESS", success_count_saved, timestamp)
                    
                elif result['piece_accuracy'] < args.success_threshold and failure_count_saved < args.num_examples:
                    failures.append(result)
                    failure_count_saved += 1
                    print(f"\n❌ FAILURE #{failure_count_saved} found! Accuracy: {result['piece_accuracy']:.3f}")
                    save_single_example_immediately(result, output_dir, args.puzzle_size, "FAILURE", failure_count_saved, timestamp)
        
        # Update progress bar with live info
        current_piece_acc = total_accuracy / total_samples if total_samples > 0 else 0
        success_count = sum(1 for r in results if r['success'])
        
        progress_bar.set_postfix({
            'piece_acc': f'{current_piece_acc:.3f}',
            'succ': success_count_saved,
            'fail': failure_count_saved,
            'total': total_samples
        })
        
        # Enhanced logging every LOG_INTERVAL samples
        if total_samples % LOG_INTERVAL == 0 and total_samples > 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            eta = (len(sample_indices) - total_samples) / samples_per_sec if samples_per_sec > 0 else 0
            
            print(f"\n📊 Progress Report (Sample {total_samples}/{len(sample_indices)})")
            print(f"   ⏱️  Elapsed: {elapsed:.1f}s | Speed: {samples_per_sec:.2f} samples/s | ETA: {eta:.1f}s")
            print(f"   🎯 Live PIECE accuracy: {current_piece_acc:.4f}")
            print(f"   🏆 Live PUZZLE accuracy: {total_perfect}/{total_samples} ({total_perfect/total_samples:.4f})")
            print(f"   🖼️  Examples saved: {success_count_saved} successes, {failure_count_saved} failures")
            print(f"   🔒 Holdout samples untouched: {len(holdout_indices)}")
        
        # Save intermediate results
        if total_samples % SAVE_INTERVAL == 0 and total_samples > 0:
            save_intermediate_results(results, output_dir, timestamp)
    
    progress_bar.close()
    
    # Calculate final statistics
    if results:
        total_samples = len(results)
        avg_accuracy = sum(r['piece_accuracy'] for r in results) / total_samples
        perfect_count = sum(r['perfect_puzzle'] for r in results)
        perfect_rate = perfect_count / total_samples
        success_count = sum(r['success'] for r in results)
        success_rate = success_count / total_samples
        total_time = time.time() - start_time
        
        print(f"\n🏁 FINAL EVALUATION RESULTS (EVALUATION SPLIT):")
        print(f"="*60)
        print(f"   📈 Total samples evaluated: {total_samples} ({args.test_split_ratio:.1%} of test set)")
        print(f"   🎯 Average PIECE accuracy: {avg_accuracy:.4f}")
        print(f"   🏆 Perfect PUZZLE rate: {perfect_count}/{total_samples} ({perfect_rate:.4f})")
        print(f"   ✅ Success rate (≥{args.success_threshold:.1%}): {success_count}/{total_samples} ({success_rate:.4f})")
        print(f"   ⏱️  Total evaluation time: {total_time:.1f}s ({total_samples/total_time:.2f} samples/s)")
        print(f"   🚀 GPU(s) used: {gpu_ids}")
        print(f"   🔒 Holdout samples for final eval: {len(holdout_indices)}")
        
        # Save detailed results
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved: {csv_path}")
        
        # Save summary with split info
        summary_path = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"{args.dataset.upper()} {args.puzzle_size}x{args.puzzle_size} Puzzle Evaluation Summary\n")
            f.write(f"="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Test split ratio: {args.test_split_ratio:.1%} for evaluation\n")
            f.write(f"Evaluation samples: {total_samples}\n")
            f.write(f"Holdout samples: {len(holdout_indices)}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"\nResults (Evaluation Split Only):\n")
            f.write(f"Average piece accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Perfect puzzle rate: {perfect_count}/{total_samples} ({perfect_rate:.4f})\n")
            f.write(f"Success rate: {success_count}/{total_samples} ({success_rate:.4f})\n")
            f.write(f"Total time: {total_time:.1f}s ({total_samples/total_time:.2f} samples/s)\n")
            f.write(f"\nNOTE: {len(holdout_indices)} samples kept for final evaluation\n")
        
        print(f"📄 Summary saved: {summary_path}")
        print(f"📄 Holdout indices saved: {holdout_path}")
        
        # Save example images
        if args.save_images:
            save_example_images(successes, failures, output_dir, args.puzzle_size)
        
    else:
        print("❌ No samples could be evaluated successfully")

if __name__ == "__main__":
    main()

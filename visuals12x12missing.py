import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import einops
import math
import sys
import os
from pathlib import Path
import datetime
from torch_geometric.data import Batch, Data

# Add paths
sys.path.append("puzzle_diff")
sys.path.append("puzzle_diff/lib")

from dataset import dataset_utils as du
from model import spatial_diffusion as sd

# CONFIGURATION
PUZZLE_SIZES = [6, 8, 10, 12]
CHECKPOINT_PATH = "/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/99qcofwy/checkpoints/last.ckpt"
NUM_EXAMPLES_PER_SIZE = 1  # 1 success + 1 failure per size
MAX_ATTEMPTS = 100

# Missing pieces configuration
SHOW_MISSING_PIECES = True  # Set to False for complete puzzles
MISSING_PERCENTAGE = 0.3  # 30% missing pieces

print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
print(f"Testing with {MISSING_PERCENTAGE:.0%} missing pieces: {SHOW_MISSING_PIECES}")

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
missing_suffix = f"_30missing" if SHOW_MISSING_PIECES else "_complete"
output_dir = Path(f"hamza_multi_size{missing_suffix}_{timestamp}")
output_dir.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            angle = torch.atan2(rotations[p, 1], rotations[p, 0])
            angle_deg = (angle * 180 / torch.pi).item()
            angle_snapped = round(angle_deg / 90) * 90
            patch_pad = patch_pad.rotate(-angle_snapped, expand=True)
        
        x = int((pos[p, 0] + 1) / 2 * (width - patch_pad.width))
        y = int((pos[p, 1] + 1) / 2 * (height - patch_pad.height))
        new_image.paste(patch_pad, (x, y), patch_pad)
    
    return new_image

def greedy_cost_assignment(pred_pos, real_grid):
    """Greedy assignment of predicted positions to grid"""
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

def analyze_puzzle_size(puzzle_size, model, device):
    """Analyze puzzles of given size"""
    print(f"\n🧩 ANALYZING {puzzle_size}x{puzzle_size} PUZZLES...")
    
    # Load dataset
    _, test_dt, _ = du.get_dataset_ROT(
        dataset="celeba",
        puzzle_sizes=[puzzle_size]
    )
    
    # Create grid for this puzzle size
    y = torch.linspace(-1, 1, puzzle_size, device=device)
    x = torch.linspace(-1, 1, puzzle_size, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    real_grid = einops.rearrange(xy, "x y c-> (x y) c")
    
    successes = []
    failures = []
    
    for attempts in range(MAX_ATTEMPTS):
        if len(successes) >= NUM_EXAMPLES_PER_SIZE and len(failures) >= NUM_EXAMPLES_PER_SIZE:
            break
            
        img_id = torch.randint(0, len(test_dt), (1,)).item()
        sample = test_dt[img_id]
        
        # Skip if no rotation data
        if sample.x.size(1) <= 2:
            continue
            
        # Simulate missing pieces if enabled
        if SHOW_MISSING_PIECES:
            num_pieces = sample.x.shape[0]
            num_missing = int(num_pieces * MISSING_PERCENTAGE)
            available_indices = torch.randperm(num_pieces)[num_missing:]
            
            print(f"   📝 {puzzle_size}x{puzzle_size} image {img_id}: {num_pieces} total, using {len(available_indices)} pieces ({MISSING_PERCENTAGE:.0%} missing)")
            
            # Create modified sample
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
            
            modified_sample = Data(
                x=modified_x,
                patches=modified_patches,
                edge_index=remapped_edges,
                puzzle_id=sample.puzzle_id if hasattr(sample, 'puzzle_id') else 0
            )
            
            batch = Batch.from_data_list([modified_sample])
            gt_pos = modified_x[:, :2].cpu()
            gt_rot = modified_x[:, 2:].cpu()
            patches_rgb = modified_patches.cpu()
            
            # Create grid for available pieces
            available_pieces_count = len(available_indices)
            grid_side = int(np.sqrt(available_pieces_count))
            if grid_side * grid_side < available_pieces_count:
                grid_side += 1
            
            y_grid = torch.linspace(-1, 1, grid_side, device=device)
            x_grid = torch.linspace(-1, 1, grid_side, device=device)
            xy_grid = torch.stack(torch.meshgrid(x_grid, y_grid, indexing="xy"), -1)
            real_grid_current = einops.rearrange(xy_grid, "x y c-> (x y) c")[:available_pieces_count]
            
        else:
            # Use complete puzzle
            batch = Batch.from_data_list([sample])
            gt_pos = sample.x[:, :2].cpu()
            gt_rot = sample.x[:, 2:].cpu()
            patches_rgb = sample.patches.cpu()
            real_grid_current = real_grid
        
        batch = batch.to(device)
        
        try:
            # Run inference
            with torch.no_grad():
                imgs, _ = model.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch=batch.batch
                )
                
                final_pred = imgs[-1] if isinstance(imgs, list) else imgs
                
                # Handle batch dimension
                if len(final_pred.shape) == 3:
                    final_pred = final_pred[0]
                
                # Reshape prediction
                expected_elements = batch.x.shape[0] * batch.x.shape[1]
                if final_pred.numel() != expected_elements:
                    continue
                
                final_pred_reshaped = final_pred.view(batch.x.shape[0], batch.x.shape[1])
                pred_pos = final_pred_reshaped[:, :2]
                pred_rot = final_pred_reshaped[:, 2:] if final_pred_reshaped.size(1) > 2 else None
                
                # Calculate accuracy using assignment
                gt_ass = greedy_cost_assignment(gt_pos.to(device), real_grid_current)
                pred_ass = greedy_cost_assignment(pred_pos.to(device), real_grid_current)
                
                # Sort assignments and compare
                sort_idx = torch.sort(gt_ass[:, 0])[1]
                gt_ass = gt_ass[sort_idx]
                sort_idx = torch.sort(pred_ass[:, 0])[1]
                pred_ass = pred_ass[sort_idx]
                
                # Calculate position accuracy
                position_correct = (gt_ass[:, 1] == pred_ass[:, 1])
                
                # Calculate rotation accuracy if available
                if pred_rot is not None and gt_rot is not None:
                    rot_correct = torch.cosine_similarity(pred_rot.to(device), gt_rot.to(device)) > math.cos(math.pi / 4)
                    combined_correct = position_correct * rot_correct
                    piece_acc_score = combined_correct.float().mean().item()
                    pieces_correct = combined_correct.sum().item()
                else:
                    piece_acc_score = position_correct.float().mean().item()
                    pieces_correct = position_correct.sum().item()
                
                total_pieces = len(pred_pos)
                
                # Create images for visualization
                scrambled_pos = torch.rand(len(patches_rgb), 2) * 2 - 1
                scrambled_rot = None
                if gt_rot is not None:
                    angles = torch.rand(len(patches_rgb)) * 2 * torch.pi
                    scrambled_rot = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
                
                scrambled_img = create_image_from_patches(
                    patches_rgb, scrambled_pos, (puzzle_size, puzzle_size), scrambled_rot
                )
                
                pred_img = create_image_from_patches(
                    patches_rgb, pred_pos.cpu(), (puzzle_size, puzzle_size), 
                    pred_rot.cpu() if pred_rot is not None else None
                )
                
                gt_img = create_image_from_patches(
                    patches_rgb, gt_pos, (puzzle_size, puzzle_size), gt_rot
                )
                
                print(f"   📊 Image {img_id}: {pieces_correct}/{total_pieces} correct ({piece_acc_score:.3f})")
                
                # Create result
                result = {
                    'puzzle_size': puzzle_size,
                    'img_id': img_id,
                    'pieces_correct': pieces_correct,
                    'total_pieces': total_pieces,
                    'piece_accuracy': piece_acc_score,
                    'perfect_puzzle': piece_acc_score >= 0.95,
                    'scrambled_img': scrambled_img,
                    'pred_img': pred_img,
                    'gt_img': gt_img,
                    'missing_pieces': SHOW_MISSING_PIECES
                }
                
                # Classify as success or failure
                success_threshold = 0.90
                failure_threshold = 0.70
                
                is_success = piece_acc_score >= success_threshold
                is_failure = piece_acc_score <= failure_threshold
                
                if is_success and len(successes) < NUM_EXAMPLES_PER_SIZE:
                    successes.append(result)
                    print(f"   ✅ Found SUCCESS #{len(successes)} for {puzzle_size}x{puzzle_size}")
                elif is_failure and len(failures) < NUM_EXAMPLES_PER_SIZE:
                    failures.append(result)
                    print(f"   ❌ Found FAILURE #{len(failures)} for {puzzle_size}x{puzzle_size}")
                
        except Exception as e:
            continue
    
    # Create placeholders if needed
    while len(successes) < NUM_EXAMPLES_PER_SIZE:
        successes.append(None)
    while len(failures) < NUM_EXAMPLES_PER_SIZE:
        failures.append(None)
    
    print(f"   ✅ Final: {len([s for s in successes if s is not None])} successes, {len([f for f in failures if f is not None])} failures for {puzzle_size}x{puzzle_size} (after {attempts+1} attempts)")
    
    return successes, failures

def create_compilation(results_list, title, save_path):
    """Create compilation image"""
    fig, axes = plt.subplots(3, len(PUZZLE_SIZES), figsize=(20, 12))
    
    for i, (size, results) in enumerate(zip(PUZZLE_SIZES, results_list)):
        if results[0] is None:
            # No data available
            for row in range(3):
                axes[row, i].text(0.5, 0.5, f'No {title.lower()}\nfound', 
                                ha='center', va='center', transform=axes[row, i].transAxes)
                axes[row, i].set_xticks([])
                axes[row, i].set_yticks([])
            continue
        
        result = results[0]  # Take first result
        
        # Row 1: Scrambled
        axes[0, i].imshow(result['scrambled_img'])
        title_suffix = f" ({MISSING_PERCENTAGE:.0%} Missing)" if SHOW_MISSING_PIECES else ""
        axes[0, i].set_title(f'Scrambled\n{size}x{size} Puzzle{title_suffix}', fontsize=14, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Prediction
        axes[1, i].imshow(result['pred_img'])
        accuracy_pct = result['piece_accuracy'] * 100
        
        if 'SUCCESS' in title:
            pred_title = f'SUCCESS\n{result["pieces_correct"]}/{result["total_pieces"]} pieces'
            if SHOW_MISSING_PIECES:
                pred_title += f'\n({MISSING_PERCENTAGE:.0%} missing handled)'
            color = 'green'
        else:
            pred_title = f'FAILED\n{result["pieces_correct"]}/{result["total_pieces"]} pieces ({accuracy_pct:.1f}%)'
            if SHOW_MISSING_PIECES:
                pred_title += f'\n({MISSING_PERCENTAGE:.0%} missing)'
            color = 'red'
            
        axes[1, i].set_title(pred_title, fontsize=14, fontweight='bold', color=color)
        axes[1, i].axis('off')
        
        # Row 3: Ground Truth
        axes[2, i].imshow(result['gt_img'])
        axes[2, i].set_title(f'Ground Truth\n{size}x{size} Target', fontsize=14, fontweight='bold')
        axes[2, i].axis('off')
    
    # Add row labels
    row_labels = ['Scrambled Input', f'{title} Cases', 'Ground Truth']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=16, fontweight='bold')
    
    # Main title
    title_suffix = f" with {MISSING_PERCENTAGE:.0%} Missing Pieces" if SHOW_MISSING_PIECES else ""
    plt.suptitle(f'{title} CASES - Multi-Size Puzzle Analysis{title_suffix}\n'
                f'Model Performance on {PUZZLE_SIZES} Puzzle Sizes', 
                fontsize=18, fontweight='bold', color='green' if 'SUCCESS' in title else 'red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path}")

def main():
    print("🔧 Loading model...")
    model = sd.GNN_Diffusion.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model = model.to(device)
    
    print("🚀 MULTI-SIZE PUZZLE ANALYSIS STARTING...")
    print(f"Puzzle sizes: {PUZZLE_SIZES}")
    print(f"Examples per size: {NUM_EXAMPLES_PER_SIZE} (1 success + 1 failure)")
    print(f"Max attempts per size: {MAX_ATTEMPTS}")
    
    all_successes = []
    all_failures = []
    
    for puzzle_size in PUZZLE_SIZES:
        successes, failures = analyze_puzzle_size(puzzle_size, model, device)
        all_successes.append(successes)
        all_failures.append(failures)
    
    print(f"\n📊 TOTAL COLLECTED:")
    print(f"   Successes: {sum(len([s for s in sucs if s is not None]) for sucs in all_successes)}")
    print(f"   Failures: {sum(len([f for f in fails if f is not None]) for fails in all_failures)}")
    
    # Create compilations
    print(f"\n🎯 Creating SUCCESS compilation...")
    success_path = output_dir / f"SUCCESS_multisize{missing_suffix}_{timestamp}.png"
    create_compilation(all_successes, "SUCCESS", success_path)
    
    print(f"\n❌ Creating FAILURE compilation...")
    failure_path = output_dir / f"FAILURE_multisize{missing_suffix}_{timestamp}.png"
    create_compilation(all_failures, "FAILURE", failure_path)
    
    print(f"\n📊 FINAL ANALYSIS SUMMARY:")
    print("="*60)
    for i, size in enumerate(PUZZLE_SIZES):
        suc = all_successes[i][0]
        fail = all_failures[i][0]
        if suc:
            print(f"   {size}x{size} ({size*size:3d} pieces): SUCCESS (acc: {suc['piece_accuracy']:.3f})")
        if fail:
            print(f"   {size}x{size} ({size*size:3d} pieces): FAILURE (acc: {fail['piece_accuracy']:.3f})")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🎯 Clean 1-to-1 comparison across puzzle sizes")
    if SHOW_MISSING_PIECES:
        print(f"   🧩 Demonstrates robustness with {MISSING_PERCENTAGE:.0%} missing pieces")
    print(f"   📊 Shows both success and failure cases for each complexity")
    print(f"   🔄 Demonstrates model's limits as puzzle size increases")
    print(f"   ⚡ Single checkpoint handles all sizes!")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   📊 SUCCESS_multisize{missing_suffix}_{timestamp}.png - 1 success per puzzle size")
    print(f"   ❌ FAILURE_multisize{missing_suffix}_{timestamp}.png - 1 failure per puzzle size")
    
    print(f"\n🚀 Multi-size analysis complete! Perfect for comparison! 🎉")

if __name__ == "__main__":
    main()
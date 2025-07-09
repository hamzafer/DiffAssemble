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
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import math
from torch_geometric.data import Batch, Data
import einops
import cv2
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Irregular Fragment Puzzle Analysis')
    parser.add_argument('--checkpoint_path', type=str, 
                       default="/home/user1/Desktop/HAMZA/THESIS/DiffAssemble/Puzzle-Diff/8fuyrpvq/checkpoints/last.ckpt",
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset to use')
    parser.add_argument('--puzzle_size', type=int, default=4, help='Puzzle size (default: 4x4)')
    parser.add_argument('--save_images', action='store_true', help='Save example images')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of success/failure examples to save (only if --save_images)')
    parser.add_argument('--full_test_set', action='store_true', help='Evaluate on full test set instead of random sampling')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to test (if not full test set)')
    parser.add_argument('--test_limit', type=int, default=100, help='Test only first N images (for quick testing)')
    parser.add_argument('--output_dir', type=str, default='imagenet_irregular_results', help='Output directory')
    parser.add_argument('--success_threshold', type=float, default=0.5, help='Accuracy threshold for success (lowered for irregular)')
    parser.add_argument('--fragment_type', type=str, choices=['regular', 'jigsaw', 'torn', 'geometric'], 
                       default='regular', help='Type of fragments to use')
    parser.add_argument('--irregular_only', action='store_true', help='Only test irregular fragments')
    parser.add_argument('--show_graph_nodes', action='store_true', help='Draw graph nodes on images')
    parser.add_argument('--erosion_percent', type=int, default=10, help='Percentage of erosion to apply to irregular fragments (0-20)')  # NEW
    
    return parser.parse_args()

def create_irregular_mask(size, fragment_type='jigsaw', erosion_percent=10):
    """Create irregular mask for fragment shapes with optional erosion"""
    h, w = size
    
    # If erosion_percent is 0, return perfect rectangle regardless of fragment_type
    if erosion_percent == 0:
        mask = np.ones((h, w), dtype=np.uint8) * 255  # Perfect rectangle
        return mask
    
    # Scale the irregularity based on erosion_percent (start much smaller)
    irregularity_scale = erosion_percent / 100.0  # 0.1 for 10%, 0.2 for 20%, etc.
    
    # Only create irregular shapes if erosion_percent > 0
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if fragment_type == 'jigsaw':
        # Create jigsaw-like edges with curves and tabs
        center_x, center_y = w // 2, h // 2
        
        # Start with basic rectangle (larger base area)
        margin = max(1, int(h * 0.15))  # 15% margin instead of 25%
        mask[margin:h-margin, margin:w-margin] = 255
        
        # Add jigsaw tabs on random sides - MUCH SMALLER initially
        base_tab_size = min(h, w) // 16  # Was //8, now //16 (half the size)
        tab_size = max(2, int(base_tab_size * irregularity_scale * 3))  # Scale with percentage
        
        # Top tab
        if np.random.rand() > 0.5:
            cv2.circle(mask, (center_x, margin), tab_size, 255, -1)
        else:
            cv2.circle(mask, (center_x, margin), tab_size, 0, -1)
            
        # Bottom tab  
        if np.random.rand() > 0.5:
            cv2.circle(mask, (center_x, h-margin), tab_size, 255, -1)
        else:
            cv2.circle(mask, (center_x, h-margin), tab_size, 0, -1)
            
        # Left tab
        if np.random.rand() > 0.5:
            cv2.circle(mask, (margin, center_y), tab_size, 255, -1)
        else:
            cv2.circle(mask, (margin, center_y), tab_size, 0, -1)
            
        # Right tab
        if np.random.rand() > 0.5:
            cv2.circle(mask, (w-margin, center_y), tab_size, 255, -1)
        else:
            cv2.circle(mask, (w-margin, center_y), tab_size, 0, -1)
            
    elif fragment_type == 'torn':
        # Create torn paper effect - MUCH SMALLER tears initially
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Scale tear intensity
        num_tears = max(1, int(2 * irregularity_scale))  # Fewer tears at low percentages
        tear_depth = max(1, int(h * 0.05 * irregularity_scale))  # Much smaller tear depth
        
        # Create random torn edges
        for _ in range(num_tears):
            # Random tear direction
            if np.random.rand() > 0.5:
                # Horizontal tear
                y_center = np.random.randint(h//3, 2*h//3)
                tear_curve = np.sin(np.linspace(0, 2*np.pi, w)) * tear_depth + y_center
                for x in range(w):
                    if np.random.rand() > (0.7 - irregularity_scale * 0.3):  # Less randomness at low percentages
                        mask[:max(0, int(tear_curve[x])), x] = 0
            else:
                # Vertical tear
                x_center = np.random.randint(w//3, 2*w//3)
                tear_curve = np.sin(np.linspace(0, 2*np.pi, h)) * tear_depth + x_center
                for y in range(h):
                    if np.random.rand() > (0.7 - irregularity_scale * 0.3):
                        mask[y, :max(0, int(tear_curve[y]))] = 0
                        
    elif fragment_type == 'geometric':
        # Create geometric irregular shapes - SMALLER and more regular initially
        shapes = ['triangle', 'pentagon', 'hexagon', 'star']
        shape = np.random.choice(shapes)
        
        center = (w//2, h//2)
        base_radius = min(h, w) // 4  # Was //3, now //4 (smaller)
        radius = max(base_radius//2, int(base_radius * (0.5 + irregularity_scale)))  # Scale with percentage
        
        if shape == 'triangle':
            pts = np.array([[center[0], center[1] - radius], 
                           [center[0] - radius, center[1] + radius//2], 
                           [center[0] + radius, center[1] + radius//2]], np.int32)
        elif shape == 'pentagon':
            angles = np.linspace(0, 2*np.pi, 6)[:-1]
            pts = np.array([[center[0] + radius * np.cos(a), 
                            center[1] + radius * np.sin(a)] for a in angles], np.int32)
        elif shape == 'hexagon':
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            pts = np.array([[center[0] + radius * np.cos(a), 
                            center[1] + radius * np.sin(a)] for a in angles], np.int32)
        elif shape == 'star':
            angles = np.linspace(0, 2*np.pi, 11)[:-1]
            pts = []
            for i, a in enumerate(angles):
                r = radius if i % 2 == 0 else radius // 2
                pts.append([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])
            pts = np.array(pts, np.int32)
            
        cv2.fillPoly(mask, [pts], 255)
    
    # Apply additional random damage/missing parts based on percentage
    if erosion_percent > 10:  # Only apply extra damage if > 10%
        damage_intensity = (erosion_percent - 10) / 90.0  # Scale to 0.0-1.0 for values 10-100
        
        # Create random damage pattern
        noise = np.random.rand(h, w)
        damage_threshold = 1.0 - damage_intensity * 0.3  # Limit damage to 30% max
        
        # Apply damage - areas with noise below threshold get removed
        damage_mask = (noise > damage_threshold).astype(np.uint8) * 255
        
        # Combine original shape with damage
        mask = cv2.bitwise_and(mask, damage_mask)
        
        # For heavy damage, add slight erosion effect
        if damage_intensity > 0.5:
            erosion_pixels = max(1, int(damage_intensity * 2))  # Reduced erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels, erosion_pixels))
            mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask

def create_irregular_fragments_from_sample(sample, fragment_type='jigsaw', erosion_percent=10):
    """Create irregular fragments from a regular puzzle sample with erosion"""
    if fragment_type == 'regular':
        return sample, None
    
    patches = sample.patches
    positions = sample.x[:, :2]
    
    irregular_patches = []
    masks = []
    
    for i, patch in enumerate(patches):
        # Convert patch to PIL Image
        if isinstance(patch, torch.Tensor):
            patch_np = (patch.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            patch_pil = Image.fromarray(patch_np)
        else:
            patch_pil = patch
        
        # Create irregular mask with erosion
        mask = create_irregular_mask(patch_pil.size, fragment_type, erosion_percent)
        mask_pil = Image.fromarray(mask).convert('L')
        
        # Apply mask to create irregular fragment
        patch_rgba = patch_pil.convert('RGBA')
        patch_rgba.putalpha(mask_pil)
        
        # Convert back to tensor with white background
        background = Image.new('RGB', patch_rgba.size, (255, 255, 255))
        background.paste(patch_rgba, mask=patch_rgba.split()[-1])
        
        transform = transforms.ToTensor()
        irregular_patch = transform(background)
        
        irregular_patches.append(irregular_patch)
        masks.append(mask)
    
    # Create new sample with irregular patches
    irregular_sample = Data(
        x=sample.x,
        patches=torch.stack(irregular_patches),
        edge_index=sample.edge_index
    )
    
    return irregular_sample, masks

def create_image_from_patches(patches, pos, n_patches, rotations=None, masks=None, show_nodes=False, edge_index=None):
    """Create puzzle image from patches and positions - SUPPORTS IRREGULAR FRAGMENTS + GRAPH NODES"""
    patch_size = 32
    height = patch_size * n_patches[0]
    width = patch_size * n_patches[1]
    new_image = Image.new("RGB", (width, height), color=(255, 255, 255))  # White background
    
    for p in range(patches.shape[0]):
        patch = patches[p, :]
        patch = Image.fromarray(
            ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
        )

        # If we have masks, apply irregular shape
        if masks is not None and p < len(masks):
            # Resize mask to match patch size
            mask_resized = cv2.resize(masks[p], (patch.width, patch.height))
            mask_pil = Image.fromarray(mask_resized).convert('L')
            
            # Create RGBA version with mask
            patch_rgba = patch.convert('RGBA')
            patch_rgba.putalpha(mask_pil)
            
            # Paste with transparency
            patch = patch_rgba
        
        # REMOVED: No more border around patches
        # patch_pad = ImageOps.expand(patch, border=1, fill=(128, 128, 128))  # Gray border
        patch_pad = patch  # Use patch directly without border
        
        if rotations is not None:
            deg_angle = (
                torch.arctan2(rotations[p, 1], rotations[p, 0]) / torch.pi * 180
            )
            deg_angle = round(deg_angle.item() / 90) * 90
            patch_pad = patch_pad.rotate(-deg_angle, fillcolor=(255, 255, 255))

        # Calculate position
        x = pos[p, 0] * (1 - 1 / n_patches[0]) 
        y = pos[p, 1] * (1 - 1 / n_patches[1])
        x_pos = int((x + 1) * width / 2) - patch_pad.width // 2
        y_pos = int((y + 1) * height / 2) - patch_pad.height // 2
        
        # Clamp positions to image bounds
        x_pos = max(0, min(x_pos, width - patch_pad.width))
        y_pos = max(0, min(y_pos, height - patch_pad.height))
        
        if patch.mode == 'RGBA':
            new_image.paste(patch_pad, (x_pos, y_pos), patch_pad)
        else:
            new_image.paste(patch_pad, (x_pos, y_pos))

    # Draw graph nodes and edges if requested
    if show_nodes:
        draw = ImageDraw.Draw(new_image)
        node_positions = []
        
        # Calculate node positions
        for p in range(patches.shape[0]):
            x = pos[p, 0] * (1 - 1 / n_patches[0]) 
            y = pos[p, 1] * (1 - 1 / n_patches[1])
            x_pos = int((x + 1) * width / 2)
            y_pos = int((y + 1) * height / 2)
            node_positions.append((x_pos, y_pos))
        
        # Draw edges first (so they appear under nodes)
        if edge_index is not None:
            for i in range(edge_index.shape[1]):
                start_node = edge_index[0, i].item()
                end_node = edge_index[1, i].item()
                if start_node < len(node_positions) and end_node < len(node_positions):
                    start_pos = node_positions[start_node]
                    end_pos = node_positions[end_node]
                    draw.line([start_pos, end_pos], fill=(255, 0, 0), width=2)  # Red edges
        
        # Draw nodes
        for i, (x_pos, y_pos) in enumerate(node_positions):
            # Draw node circle
            radius = 8
            draw.ellipse([x_pos-radius, y_pos-radius, x_pos+radius, y_pos+radius], 
                        fill=(0, 255, 0), outline=(0, 0, 0), width=2)  # Green nodes with black outline
            
            # Draw node number
            bbox = draw.textbbox((0, 0), str(i))
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x_pos - text_width//2, y_pos - text_height//2), str(i), 
                     fill=(0, 0, 0))  # Black text

    return new_image

def create_original_image_from_sample(sample, puzzle_size):
    """Create the original image from a sample before making it irregular"""
    patches = sample.patches
    positions = sample.x[:, :2]
    
    # Create the original assembled image
    original_img = create_image_from_patches(
        patches, positions, (puzzle_size, puzzle_size), 
        None, None, show_nodes=False, edge_index=None
    )
    
    return original_img

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

def evaluate_single_sample(model, sample, real_grid, device, img_id, fragment_type='regular', erosion_percent=10):
    """Evaluate a single puzzle sample with irregular fragments"""
    try:
        # Create irregular version if needed
        irregular_sample, masks = create_irregular_fragments_from_sample(sample, fragment_type, erosion_percent)
        
        # Create batch
        batch = Batch.from_data_list([irregular_sample])
        batch = batch.to(device)
        
        # Extract ground truth - POSITION-ONLY
        gt_pos = irregular_sample.x[:, :2].cpu()
        patches_rgb = irregular_sample.patches.cpu()
        
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
        expected_total_elements = irregular_sample.x.shape[0] * irregular_sample.x.shape[1]
        if final_pred.numel() != expected_total_elements:
            return None
        
        # Reshape
        final_pred = final_pred.view(irregular_sample.x.shape[0], irregular_sample.x.shape[1])
        pred_pos = final_pred[:, :2]
        
        # For irregular fragments, use distance-based accuracy instead of exact assignment
        if fragment_type != 'regular':
            # Calculate position accuracy based on distance
            pos_distances = torch.norm(gt_pos - pred_pos, dim=1)
            threshold = 0.3  # Acceptable position error for irregular fragments
            correct_positions = (pos_distances < threshold).float()
            
            piece_accuracy = correct_positions
            piece_acc_score = piece_accuracy.mean().item()
            pieces_correct = piece_accuracy.sum().int().item()
            total_pieces = irregular_sample.x.shape[0]
            perfect_puzzle = correct_positions.all().item()
        else:
            # Original grid-based accuracy for regular fragments
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
            total_pieces = irregular_sample.x.shape[0]
            perfect_puzzle = position_correct.all().item()
        
        result = {
            'img_id': img_id,
            'total_pieces': total_pieces,
            'pieces_correct': pieces_correct,
            'piece_accuracy': piece_acc_score,
            'perfect_puzzle': perfect_puzzle,
            'gt_pos': gt_pos,
            'pred_pos': pred_pos,
            'patches_rgb': patches_rgb,
            'masks': masks,
            'fragment_type': fragment_type,
            'edge_index': irregular_sample.edge_index.cpu(),  # Store edge_index for graph viz
            'original_sample': sample  # Store original sample for comparison
        }
        
        return result
        
    except Exception as e:
        print(f"Error evaluating sample {img_id}: {e}")
        return None

def save_example_images(successes, failures, output_dir, puzzle_size, fragment_type):
    """Save example success and failure images for irregular fragments"""
    if not successes and not failures:
        print(f"   ⚠️  No examples to save for {fragment_type}")
        return
    
    print(f"💾 Saving {fragment_type} example images...")
    
    # Take up to 3 examples each
    success_examples = successes[:3] if successes else []
    failure_examples = failures[:3] if failures else []
    
    # Combine all results
    all_results = []
    if success_examples:
        all_results.extend([(s, "SUCCESS") for s in success_examples])
    if failure_examples:
        all_results.extend([(f, "FAILURE") for f in failure_examples])
    
    if not all_results:
        print(f"   ⚠️  No valid examples for {fragment_type}")
        return
    
    num_cols = len(all_results)
    fig, axes = plt.subplots(3, num_cols, figsize=(5*num_cols, 15))
    
    # Handle single column case
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (result, status) in enumerate(all_results):
        try:
            # Create scrambled position for visualization
            scrambled_pos = torch.rand(puzzle_size*puzzle_size, 2) * 2 - 1
            
            # Create images
            scrambled_img = create_image_from_patches(
                result['patches_rgb'], scrambled_pos, (puzzle_size, puzzle_size), 
                None, result['masks']
            )
            pred_img = create_image_from_patches(
                result['patches_rgb'], result['pred_pos'], (puzzle_size, puzzle_size), 
                None, result['masks']
            )
            gt_img = create_image_from_patches(
                result['patches_rgb'], result['gt_pos'], (puzzle_size, puzzle_size), 
                None, result['masks']
            )
            
            # Convert RGBA to RGB for display
            if scrambled_img.mode == 'RGBA':
                bg = Image.new('RGB', scrambled_img.size, (255, 255, 255))
                bg.paste(scrambled_img, mask=scrambled_img.split()[-1])
                scrambled_img = bg
                
            if pred_img.mode == 'RGBA':
                bg = Image.new('RGB', pred_img.size, (255, 255, 255))
                bg.paste(pred_img, mask=pred_img.split()[-1])
                pred_img = bg
                
            if gt_img.mode == 'RGBA':
                bg = Image.new('RGB', gt_img.size, (255, 255, 255))
                bg.paste(gt_img, mask=gt_img.split()[-1])
                gt_img = bg
            
            # Row 1: Scrambled
            axes[0, i].imshow(scrambled_img)
            axes[0, i].set_title(f'Scrambled\n{fragment_type.title()}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Row 2: Prediction
            axes[1, i].imshow(pred_img)
            color = 'green' if status == "SUCCESS" else 'red'
            accuracy_pct = result['piece_accuracy'] * 100
            axes[1, i].set_title(f'{status}\n{result["pieces_correct"]}/{result["total_pieces"]} ({accuracy_pct:.1f}%)', 
                               fontsize=12, fontweight='bold', color=color)
            axes[1, i].axis('off')
            
            # Row 3: Ground Truth
            axes[2, i].imshow(gt_img)
            axes[2, i].set_title(f'Ground Truth\nID: {result["img_id"]}', fontsize=12, fontweight='bold')
            axes[2, i].axis('off')
            
        except Exception as e:
            print(f"   ⚠️  Error creating visualization for example {i}: {e}")
            # Fill with empty plots
            for row in range(3):
                axes[row, i].axis('off')
                axes[row, i].text(0.5, 0.5, f'Error\nExample {i}', 
                                ha='center', va='center', transform=axes[row, i].transAxes)
    
    title = f'{puzzle_size}x{puzzle_size} {fragment_type.title()} Fragment Puzzle Assembly'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f"{fragment_type}_{puzzle_size}x{puzzle_size}_examples.png"
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Examples saved: {save_path}")
    except Exception as e:
        print(f"   ❌ Failed to save {fragment_type} examples: {e}")
        plt.close()

def save_single_example(result, output_dir, puzzle_size, fragment_type, status, example_num, show_nodes=False):
    """Save a single example immediately when found"""
    try:
        print(f"   💾 Saving {status.lower()} example {example_num}...")
        
        # Create images
        scrambled_pos = torch.rand(puzzle_size*puzzle_size, 2) * 2 - 1
        
        # Get edge_index for graph visualization
        edge_index = result.get('edge_index', None)
        
        # Create original image (before irregular transformation)
        original_img = create_original_image_from_sample(result['original_sample'], puzzle_size)
        
        scrambled_img = create_image_from_patches(
            result['patches_rgb'], scrambled_pos, (puzzle_size, puzzle_size), 
            None, result['masks'], show_nodes, edge_index
        )
        pred_img = create_image_from_patches(
            result['patches_rgb'], result['pred_pos'], (puzzle_size, puzzle_size), 
            None, result['masks'], show_nodes, edge_index
        )
        gt_img = create_image_from_patches(
            result['patches_rgb'], result['gt_pos'], (puzzle_size, puzzle_size), 
            None, result['masks'], show_nodes, edge_index
        )
        
        # Convert RGBA to RGB for display
        for img in [scrambled_img, pred_img, gt_img]:
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
        
        # Create figure with 4 columns now (original + 3 others)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Image\n(Before Irregular)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Scrambled
        axes[1].imshow(scrambled_img)
        axes[1].set_title(f'Scrambled\n{fragment_type.title()}', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_img)
        color = 'green' if status == "SUCCESS" else 'red'
        accuracy_pct = result['piece_accuracy'] * 100
        axes[2].set_title(f'{status}\n{result["pieces_correct"]}/{result["total_pieces"]} ({accuracy_pct:.1f}%)', 
                         fontsize=12, fontweight='bold', color=color)
        axes[2].axis('off')
        
        # Ground Truth
        axes[3].imshow(gt_img)
        axes[3].set_title(f'Ground Truth\nID: {result["img_id"]}', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        title = f'{fragment_type.title()} Fragment Example - {status}'
        if show_nodes:
            title += ' (with Graph Nodes)'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save immediately
        nodes_suffix = "_with_nodes" if show_nodes else ""
        save_path = output_dir / f"{fragment_type}_{status.lower()}_example_{example_num}_img_{result['img_id']}{nodes_suffix}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Saved: {save_path}")
        
    except Exception as e:
        print(f"      ❌ Failed to save example: {e}")

def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fragment_types = ['regular', 'jigsaw', 'torn', 'geometric'] if args.fragment_type == 'all' else [args.fragment_type]
    if args.irregular_only:
        fragment_types = ['jigsaw', 'torn', 'geometric']
    
    print(f"🧩 IRREGULAR FRAGMENT PUZZLE ANALYSIS")
    print(f"Fragment types: {fragment_types}")
    print(f"Puzzle size: {args.puzzle_size}x{args.puzzle_size}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Test limit: {args.test_limit} images")  # NEW
    print(f"Show graph nodes: {args.show_graph_nodes}")  # NEW
    
    # Load model
    print("\n🔧 Loading model...")
    try:
        model = sd.GNN_Diffusion.load_from_checkpoint(args.checkpoint_path)
        model.initialize_torchmetrics([args.puzzle_size])
        model.noise_weight = 0.0
        model.inference_ratio = 10
        model.save_eval_images = False
        model = model.to(device)
        model.eval()
        print("   ✅ Model loaded successfully")
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
    
    # Determine samples to test - USE TEST LIMIT
    if args.full_test_set:
        sample_indices = list(range(min(args.test_limit, len(test_dt))))  # MODIFIED
    else:
        max_samples = min(args.max_samples, len(test_dt), args.test_limit)  # MODIFIED
        sample_indices = list(range(max_samples))  # Use first N images instead of random
    
    print(f"   🎯 Testing {len(sample_indices)} samples")
    
    # Test each fragment type
    all_results = {}
    
    for fragment_type in fragment_types:
        print(f"\n🔍 Testing {fragment_type} fragments...")
        
        results = []
        successes = []
        failures = []
        
        total_accuracy = 0
        total_perfect = 0
        total_samples = 0
        
        progress_bar = tqdm(sample_indices, desc=f"Evaluating {fragment_type}", unit="sample")
        
        for img_id in progress_bar:
            sample = test_dt[img_id]
            result = evaluate_single_sample(model, sample, real_grid, device, img_id, fragment_type, args.erosion_percent)
            if result is not None:
                results.append({
                    'img_id': result['img_id'],
                    'total_pieces': result['total_pieces'],
                    'pieces_correct': result['pieces_correct'],
                    'piece_accuracy': result['piece_accuracy'],
                    'perfect_puzzle': result['perfect_puzzle'],
                    'success': result['piece_accuracy'] >= args.success_threshold,
                    'fragment_type': fragment_type
                })
                
                total_accuracy += result['piece_accuracy']
                total_perfect += int(result['perfect_puzzle'])
                total_samples += 1
                
                # SAVE IMAGES IMMEDIATELY when found
                if args.save_images:
                    if result['piece_accuracy'] >= args.success_threshold and len(successes) < args.num_examples:
                        print(f"   [SUCCESS] Example collected (img_id={result['img_id']}, acc={result['piece_accuracy']:.3f})")
                        successes.append(result)
                        # SAVE THIS SUCCESS EXAMPLE RIGHT NOW
                        save_single_example(result, output_dir, args.puzzle_size, fragment_type, "SUCCESS", len(successes), args.show_graph_nodes)
                        
                    elif result['piece_accuracy'] < args.success_threshold and len(failures) < args.num_examples:
                        print(f"   [FAILURE] Example collected (img_id={result['img_id']}, acc={result['piece_accuracy']:.3f})")
                        failures.append(result)
                        # SAVE THIS FAILURE EXAMPLE RIGHT NOW
                        save_single_example(result, output_dir, args.puzzle_size, fragment_type, "FAILURE", len(failures), args.show_graph_nodes)
                
                # Update progress bar
                current_avg_acc = total_accuracy / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix({
                    'avg_acc': f'{current_avg_acc:.3f}',
                    'perfect': f'{total_perfect}/{total_samples}'
                })
        
        progress_bar.close()
        
        # Store results for this fragment type
        if total_samples > 0:
            avg_accuracy = total_accuracy / total_samples
            perfect_rate = total_perfect / total_samples
            success_count = len([r for r in results if r['success']])
            success_rate = success_count / total_samples
            
            all_results[fragment_type] = {
                'avg_accuracy': avg_accuracy,
                'perfect_rate': perfect_rate,
                'success_rate': success_rate,
                'total_samples': total_samples,
                'successes': successes,
                'failures': failures,
                'results': results
            }
            
            print(f"   📊 {fragment_type.title()} Results:")
            print(f"      🎯 Average accuracy: {avg_accuracy:.4f}")
            print(f"      🏆 Perfect puzzles: {total_perfect}/{total_samples} ({perfect_rate:.4f})")
            print(f"      ✅ Success rate: {success_count}/{total_samples} ({success_rate:.4f})")
        else:
            print(f"   ❌ No successful evaluations for {fragment_type}")
    
    # Save results and comparisons
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        all_detailed_results = []
        for fragment_type, data in all_results.items():
            all_detailed_results.extend(data['results'])
        
        csv_path = output_dir / f"irregular_evaluation_results_{timestamp}.csv"
        df = pd.DataFrame(all_detailed_results)
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved: {csv_path}")
        
        # Save comparison summary
        summary_path = output_dir / f"irregular_evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Irregular Fragment Puzzle Evaluation Summary\n")
            f.write(f"="*60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Puzzle size: {args.puzzle_size}x{args.puzzle_size}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Test limit: {args.test_limit} images\n")
            f.write(f"Success threshold: {args.success_threshold:.1%}\n")
            f.write(f"Graph nodes shown: {args.show_graph_nodes}\n\n")
            
            f.write("Results by Fragment Type:\n")
            f.write("-" * 40 + "\n")
            for fragment_type, data in all_results.items():
                f.write(f"{fragment_type.title()} Fragments:\n")
                f.write(f"  Average accuracy: {data['avg_accuracy']:.4f}\n")
                f.write(f"  Perfect puzzles: {data['perfect_rate']:.4f}\n")
                f.write(f"  Success rate: {data['success_rate']:.4f}\n")
                f.write(f"  Total samples: {data['total_samples']}\n\n")
        
        print(f"📄 Summary saved: {summary_path}")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FeatureVisualizer:
    """Visualize extracted DINOv3 features"""
    
    def __init__(self, feature_dir: Union[str, Path]):
        self.feature_dir = Path(feature_dir)
        self.features = self._load_features()
        
    def _load_features(self) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Load all feature files from directory"""
        features = {}
        
        for file_path in self.feature_dir.glob("*.npy"):
            feature_name = file_path.stem
            features[feature_name] = np.load(file_path)
            print(f"Loaded {feature_name}: {features[feature_name].shape}")
        
        for file_path in self.feature_dir.glob("*.npz"):
            feature_name = file_path.stem
            npz_data = np.load(file_path)
            features[feature_name] = {key: npz_data[key] for key in npz_data.keys()}
            print(f"Loaded {feature_name} with {len(features[feature_name])} images")
        
        return features
    
    def visualize_patch_features(self, 
                                feature_key: str = "patch_features_layer_-1",
                                image_name: Optional[str] = None,
                                method: str = "pca",
                                output_size: Tuple[int, int] = (224, 224),
                                save_path: Optional[str] = None,
                                save_individual: bool = False):
        """Visualize patch features as RGB pseudocolor map following DINOv3 visualization"""
        
        if feature_key not in self.features:
            print(f"Feature {feature_key} not found. Available: {list(self.features.keys())}")
            return
        
        patch_features = self.features[feature_key]
        
        # Handle single image or multiple images
        if isinstance(patch_features, dict):
            if image_name is None:
                image_name = list(patch_features.keys())[0]
            if image_name not in patch_features:
                print(f"Image {image_name} not found. Available: {list(patch_features.keys())}")
                return
            features = patch_features[image_name]
        else:
            features = patch_features
        
        print(f"Visualizing {feature_key} for {image_name or 'single image'}")
        print(f"Feature shape: {features.shape}")
        
        # Step 1: Reshape patch features to spatial grid [H_patches, W_patches, embed_dim]
        num_patches = features.shape[0]
        embed_dim = features.shape[1]
        
        # Calculate patch grid dimensions
        patch_size = int(np.sqrt(num_patches))
        if patch_size * patch_size != num_patches:
            # Handle non-square patch grids
            for i in range(int(np.sqrt(num_patches)), 0, -1):
                if num_patches % i == 0:
                    h_patches = i
                    w_patches = num_patches // i
                    break
        else:
            h_patches = w_patches = patch_size
        
        print(f"Patch grid: {h_patches} x {w_patches} = {num_patches} patches")
        
        # Reshape to spatial grid
        features_spatial = features.reshape(h_patches, w_patches, embed_dim)
        
        # Step 2: Apply PCA to reduce embedding dimension to 3 (RGB)
        # Keep spatial structure, only reduce feature dimension
        features_flat = features_spatial.reshape(-1, embed_dim)  # [H*W, embed_dim]
        
        if method == "pca":
            reducer = PCA(n_components=3)
            features_rgb = reducer.fit_transform(features_flat)  # [H*W, 3]
            print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_[:3]}")
            print(f"Total variance explained: {reducer.explained_variance_ratio_[:3].sum():.3f}")
        elif method == "tsne":
            print("Applying t-SNE (this may take a moment)...")
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, num_patches-1))
            features_rgb = reducer.fit_transform(features_flat)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Step 3: Reshape back to spatial grid [H_patches, W_patches, 3]
        features_rgb_spatial = features_rgb.reshape(h_patches, w_patches, 3)
        
        # Step 4: Normalize to [0, 1] for RGB visualization
        features_rgb_norm = (features_rgb_spatial - features_rgb_spatial.min()) / \
                           (features_rgb_spatial.max() - features_rgb_spatial.min())
        
        # Step 5: Resize to target image size using interpolation
        try:
            from scipy.ndimage import zoom
        except ImportError:
            print("scipy not found, installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'scipy'])
            from scipy.ndimage import zoom
        zoom_h = output_size[0] / h_patches
        zoom_w = output_size[1] / w_patches
        
        # Resize each RGB channel separately
        features_resized = np.zeros((output_size[0], output_size[1], 3))
        for c in range(3):
            features_resized[:, :, c] = zoom(features_rgb_norm[:, :, c], 
                                           (zoom_h, zoom_w), order=1)
        
        # Calculate feature magnitude map for comparison
        feature_norms = np.linalg.norm(features_spatial, axis=2)
        feature_norms_resized = zoom(feature_norms, (zoom_h, zoom_w), order=1)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original patch resolution
        plt.subplot(1, 3, 1)
        plt.imshow(features_rgb_norm)
        plt.title(f'{method.upper()} RGB Visualization\nPatch Resolution ({h_patches}x{w_patches})')
        plt.axis('off')
        
        # Resized to image resolution  
        plt.subplot(1, 3, 2)
        plt.imshow(features_resized)
        plt.title(f'Resized to {output_size[0]}x{output_size[1]}\nRGB Pseudocolor Map')
        plt.axis('off')
        
        # Feature magnitude heatmap
        plt.subplot(1, 3, 3)
        plt.imshow(feature_norms_resized, cmap='viridis')
        plt.colorbar(label='Feature Magnitude', shrink=0.7)
        plt.title('Feature Magnitude Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            if save_individual:
                save_dir = Path(save_path).parent
                save_stem = Path(save_path).stem

                plt.imsave(save_dir / f"{save_stem}_patch_rgb.png", features_rgb_norm)
                plt.imsave(save_dir / f"{save_stem}_resized_rgb.png", features_resized)
                plt.imsave(save_dir / f"{save_stem}_magnitude.png", feature_norms_resized, cmap='viridis')
                print(f"Saved individual visualizations to {save_dir}")
        else:
            plt.show()
        
        return features_resized, feature_norms_resized
    
    def visualize_global_features(self,
                                 feature_key: str = "global_features",
                                 method: str = "pca",
                                 save_path: Optional[str] = None):
        """Visualize global features analysis and dimensionality reduction"""
        
        if feature_key not in self.features:
            print(f"Feature {feature_key} not found. Available: {list(self.features.keys())}")
            return
        
        global_features = self.features[feature_key]
        
        if isinstance(global_features, dict):
            # Multiple images - show clustering
            image_names = list(global_features.keys())
            features_array = np.stack([global_features[name] for name in image_names])
            
            plt.figure(figsize=(15, 5))
            
            if method == "pca":
                reducer = PCA(n_components=2)
                reduced_features = reducer.fit_transform(features_array)
                print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_[:2]}")
            elif method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
                reduced_features = reducer.fit_transform(features_array)
            
            # 2D projection
            plt.subplot(1, 3, 1)
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                c=range(len(image_names)), cmap='tab10', s=100)
            for i, name in enumerate(image_names):
                plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.title(f'{method.upper()} 2D Projection\n({len(image_names)} images)')
            plt.xlabel(f'{method.upper()} 1')
            plt.ylabel(f'{method.upper()} 2')
            
            # Feature similarity heatmap
            plt.subplot(1, 3, 2)
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(features_array)
            im = plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, label='Cosine Similarity')
            plt.title('Feature Similarity Matrix')
            plt.xticks(range(len(image_names)), image_names, rotation=45)
            plt.yticks(range(len(image_names)), image_names)
            
            # Average feature statistics
            plt.subplot(1, 3, 3)
            mean_features = np.mean(features_array, axis=0)
            std_features = np.std(features_array, axis=0)
            plt.fill_between(range(len(mean_features)), 
                           mean_features - std_features, 
                           mean_features + std_features, alpha=0.3)
            plt.plot(mean_features, label='Mean')
            plt.title('Average Global Features\n(across all images)')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Feature Value')
            plt.legend()
            
        else:
            # Single image - comprehensive analysis
            print(f"Global feature shape: {global_features.shape}")
            print(f"Feature range: [{global_features.min():.4f}, {global_features.max():.4f}]")
            print(f"Feature norm: {np.linalg.norm(global_features):.4f}")
            print(f"Sparsity: {(np.abs(global_features) < 1e-6).sum()}/{len(global_features)} zero values")
            
            plt.figure(figsize=(18, 10))
            
            # Feature value distribution
            plt.subplot(2, 4, 1)
            plt.hist(global_features.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Feature Value Distribution')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Feature values by dimension
            plt.subplot(2, 4, 2)
            plt.plot(global_features.flatten(), linewidth=0.8)
            plt.title('Feature Values by Dimension')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Feature Value')
            plt.grid(True, alpha=0.3)
            
            # Top-k largest absolute values
            plt.subplot(2, 4, 3)
            top_k = 20
            top_indices = np.argsort(np.abs(global_features.flatten()))[-top_k:]
            top_values = global_features.flatten()[top_indices]
            colors = ['red' if v > 0 else 'blue' for v in top_values]
            plt.barh(range(len(top_values)), top_values, color=colors)
            plt.title(f'Top {top_k} Features (by magnitude)')
            plt.xlabel('Feature Value')
            plt.ylabel('Rank')
            
            # Feature statistics
            plt.subplot(2, 4, 4)
            stats = {
                'Mean': np.mean(global_features),
                'Std': np.std(global_features),
                'Min': np.min(global_features),
                'Max': np.max(global_features),
                'L2 Norm': np.linalg.norm(global_features),
                'L1 Norm': np.linalg.norm(global_features, ord=1)
            }
            bars = plt.bar(range(len(stats)), list(stats.values()))
            plt.xticks(range(len(stats)), list(stats.keys()), rotation=45)
            plt.title('Feature Statistics')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 2D visualization using first 2 principal components
            plt.subplot(2, 4, 5)
            if len(global_features) >= 2:
                # Reshape to patches-like for PCA visualization
                patch_size = int(np.sqrt(len(global_features) / 16))  # Approximate 16:1 ratio
                if patch_size == 0:
                    patch_size = 1
                    
                # Try different reshape strategies
                reshaped_for_pca = None
                for h in range(1, int(np.sqrt(len(global_features))) + 1):
                    if len(global_features) % h == 0:
                        w = len(global_features) // h
                        reshaped_for_pca = global_features.reshape(h, w)
                        break
                
                if reshaped_for_pca is not None:
                    plt.imshow(reshaped_for_pca, cmap='RdBu_r', aspect='auto')
                    plt.colorbar(label='Feature Value', shrink=0.7)
                    plt.title(f'Feature Map\n({reshaped_for_pca.shape[0]}x{reshaped_for_pca.shape[1]})')
                else:
                    plt.plot(global_features)
                    plt.title('Unable to reshape\nShowing 1D plot')
            else:
                plt.text(0.5, 0.5, 'Not enough features\nfor visualization', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Feature Visualization')
            
            # Cumulative feature contribution
            plt.subplot(2, 4, 6)
            sorted_abs_features = np.sort(np.abs(global_features.flatten()))[::-1]
            cumsum = np.cumsum(sorted_abs_features)
            cumsum_normalized = cumsum / cumsum[-1]
            plt.plot(cumsum_normalized)
            plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%')
            plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95%')
            plt.title('Cumulative Feature Contribution')
            plt.xlabel('Feature Rank')
            plt.ylabel('Cumulative Contribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Feature magnitude heatmap (reshaped)
            plt.subplot(2, 4, 7)
            # Create a square-ish heatmap
            side_length = int(np.sqrt(len(global_features)))
            if side_length * side_length <= len(global_features):
                padded_features = np.pad(global_features, 
                                       (0, side_length * side_length - len(global_features)), 
                                       'constant', constant_values=0)
                feature_heatmap = padded_features.reshape(side_length, side_length)
            else:
                feature_heatmap = global_features[:side_length*side_length].reshape(side_length, side_length)
                
            plt.imshow(np.abs(feature_heatmap), cmap='viridis')
            plt.colorbar(label='|Feature Value|', shrink=0.7)
            plt.title('Feature Magnitude Heatmap')
            
            # PCA analysis on feature structure
            plt.subplot(2, 4, 8)
            if len(global_features) > 10:
                # Sliding window analysis to see feature structure
                window_size = min(50, len(global_features) // 10)
                if window_size > 1:
                    windowed_features = []
                    for i in range(0, len(global_features) - window_size + 1, window_size // 2):
                        windowed_features.append(global_features[i:i + window_size])
                    
                    if len(windowed_features) > 1:
                        windowed_features = np.array(windowed_features)
                        pca = PCA(n_components=min(2, windowed_features.shape[0]))
                        reduced = pca.fit_transform(windowed_features)
                        
                        if reduced.shape[1] >= 2:
                            plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)), cmap='viridis')
                            plt.title(f'Feature Structure PCA\n(window size: {window_size})')
                            plt.xlabel('PC1')
                            plt.ylabel('PC2')
                        else:
                            plt.plot(reduced[:, 0])
                            plt.title('Feature Structure (1D)')
                    else:
                        plt.plot(global_features)
                        plt.title('Feature Values')
                else:
                    plt.plot(global_features)
                    plt.title('All Feature Values')
            else:
                plt.bar(range(len(global_features)), global_features)
                plt.title('Individual Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def compare_features(self, 
                        feature_keys: List[str],
                        image_name: Optional[str] = None,
                        save_path: Optional[str] = None):
        """Compare different types of features"""
        
        fig, axes = plt.subplots(2, len(feature_keys), figsize=(4*len(feature_keys), 8))
        if len(feature_keys) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature_key in enumerate(feature_keys):
            if feature_key not in self.features:
                print(f"Feature {feature_key} not found")
                continue
            
            features = self.features[feature_key]
            
            # Handle dict vs array
            if isinstance(features, dict):
                if image_name is None:
                    image_name = list(features.keys())[0]
                feature_data = features[image_name]
            else:
                feature_data = features
            
            # Feature distribution
            axes[0, i].hist(feature_data.flatten(), bins=50, alpha=0.7)
            axes[0, i].set_title(f'{feature_key}\nDistribution')
            axes[0, i].set_xlabel('Feature Value')
            axes[0, i].set_ylabel('Frequency')
            
            # Feature statistics
            stats = {
                'Mean': np.mean(feature_data),
                'Std': np.std(feature_data),
                'Min': np.min(feature_data),
                'Max': np.max(feature_data),
                'Norm': np.linalg.norm(feature_data)
            }
            
            axes[1, i].bar(stats.keys(), stats.values())
            axes[1, i].set_title(f'{feature_key}\nStatistics')
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        else:
            plt.show()
    
    def print_summary(self):
        """Print summary of loaded features"""
        print("\n=== Feature Summary ===")
        for feature_key, feature_data in self.features.items():
            if isinstance(feature_data, dict):
                print(f"\n{feature_key}:")
                print(f"  Type: Multiple images ({len(feature_data)} images)")
                sample_shape = list(feature_data.values())[0].shape
                print(f"  Shape per image: {sample_shape}")
                print(f"  Images: {list(feature_data.keys())}")
            else:
                print(f"\n{feature_key}:")
                print(f"  Type: Single image")
                print(f"  Shape: {feature_data.shape}")
                print(f"  Dtype: {feature_data.dtype}")
                print(f"  Range: [{feature_data.min():.3f}, {feature_data.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features")
    parser.add_argument("feature_dir", type=str, help="Directory containing feature files")
    parser.add_argument("--patch-features", action="store_true", help="Visualize patch features")
    parser.add_argument("--global-features", action="store_true", help="Visualize global features")
    parser.add_argument("--compare", action="store_true", help="Compare different features")
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="pca",
                       help="Dimensionality reduction method")
    parser.add_argument("--image-name", type=str, help="Specific image name to visualize")
    parser.add_argument("--save-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--save-individual", action="store_true",
                       help="Also save patch RGB, resized RGB, and magnitude maps as separate PNG files")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = FeatureVisualizer(args.feature_dir)
    visualizer.print_summary()
    
    # Create save directory if specified
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize patch features
    if args.patch_features:
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / "patch_features.png"
        visualizer.visualize_patch_features(
            method=args.method,
            image_name=args.image_name,
            save_path=save_path,
            save_individual=args.save_individual,
        )
    
    # Visualize global features
    if args.global_features:
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / "global_features.png"
        visualizer.visualize_global_features(
            method=args.method,
            save_path=save_path
        )
    
    # Compare features
    if args.compare:
        feature_keys = list(visualizer.features.keys())
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / "feature_comparison.png"
        visualizer.compare_features(
            feature_keys=feature_keys,
            image_name=args.image_name,
            save_path=save_path
        )
    
    # If no specific visualization requested, show summary and basic viz
    if not any([args.patch_features, args.global_features, args.compare]):
        print("\nNo specific visualization requested. Use --patch-features, --global-features, or --compare")
        print("Example usage:")
        print(f"  python {Path(__file__).name} {args.feature_dir} --patch-features --method pca")
        print(f"  python {Path(__file__).name} {args.feature_dir} --global-features --method tsne")
        print(f"  python {Path(__file__).name} {args.feature_dir} --compare")


if __name__ == "__main__":
    main()

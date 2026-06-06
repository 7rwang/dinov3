#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import v2

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Image processing configuration"""
    resize_size: int = 224
    crop_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass  
class FeatureConfig:
    """Feature extraction configuration"""
    layers: List[int] = field(default_factory=lambda: [-1])  # Which layers to extract from (-1 means last layer)
    use_cls_token: bool = True  # Whether to use CLS token for global features
    normalize: bool = True  # Whether to normalize features
    patch_features: bool = True  # Whether to extract patch-level features


@dataclass
class VisualizationConfig:
    """Feature visualization configuration"""
    enabled: bool = True
    patch_features: bool = True
    global_features: bool = False
    compare: bool = False
    save_individual: bool = True
    method: str = "pca"
    output_size: Union[List[int], str] = "input"
    save_dir: str = "visualizations"


@dataclass
class ModelConfig:
    """Model configuration"""
    dino_hub: Optional[str] = None  # DINOv3 model name from torch.hub
    config_file: Optional[str] = None
    pretrained_weights: Optional[str] = None
    local_model_path: Optional[str] = None  # Path to local model directory


@dataclass
class ExtractionConfig:
    """Feature extraction pipeline configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    input_path: str = "input_images"  # Input image path or directory
    output_path: str = "features"  # Output directory for features
    batch_size: int = 8  # Batch size for processing multiple images
    device: str = "cuda"  # Device to use
    save_format: str = "npz"  # Output format: 'npz', 'pt', 'h5'


class ImageProcessor:
    """Image preprocessing pipeline"""
    
    def __init__(self, config: ImageConfig):
        self.config = config
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create image preprocessing transform"""
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.config.resize_size, self.config.resize_size), antialias=True),
            v2.CenterCrop((self.config.crop_size, self.config.crop_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.config.mean, std=self.config.std),
        ])
        return transform
    
    def process_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Process a single image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def process_batch(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Process a batch of images"""
        batch = []
        for image_path in image_paths:
            image_tensor = self.process_image(image_path)
            batch.append(image_tensor)
        return torch.stack(batch)


class FeatureExtractor:
    """DINOv3 feature extraction pipeline"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model 
        if config.model.local_model_path:
            # Load from local path using transformers
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(config.model.local_model_path)
            self.model = AutoModel.from_pretrained(config.model.local_model_path)
            self.model_context = {'autocast_dtype': torch.float}
        elif config.model.dino_hub:
            self.model = torch.hub.load('facebookresearch/dinov3', config.model.dino_hub, source='github')
            self.model_context = {'autocast_dtype': torch.float}
        else:
            raise ValueError("Please specify either model.local_model_path or model.dino_hub")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup image processor
        self.image_processor = ImageProcessor(config.image)
        
        logger.info(f"Loaded model on device: {self.device}")
        logger.info(f"Model type: {type(self.model).__name__}")

    def _split_transformers_tokens(self, hidden_state: torch.Tensor, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split HuggingFace DINOv3 output into CLS and patch tokens."""
        cls_token = hidden_state[:, 0]

        patch_size = getattr(getattr(self.model, "config", None), "patch_size", None)
        if patch_size is None:
            patch_size = getattr(self.model, "patch_size", None)
        if patch_size is None:
            raise ValueError("Cannot infer patch_size for transformer model output")

        h_patches = images.shape[-2] // int(patch_size)
        w_patches = images.shape[-1] // int(patch_size)
        num_patch_tokens = h_patches * w_patches
        num_extra_tokens = hidden_state.shape[1] - 1 - num_patch_tokens
        if num_extra_tokens < 0:
            raise ValueError(
                f"Model returned {hidden_state.shape[1]} tokens, fewer than expected "
                f"1 CLS + {num_patch_tokens} patch tokens"
            )

        patch_tokens = hidden_state[:, 1 + num_extra_tokens:]
        return cls_token, patch_tokens

    def _extract_transformers_features(self, images: torch.Tensor) -> dict:
        """Extract configured layers from a HuggingFace DINOv3 model."""
        with torch.autocast(device_type=self.device.type, dtype=self.model_context['autocast_dtype']):
            outputs = self.model(images, output_hidden_states=True)

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("Transformer model did not return hidden_states")

        features = {}
        for layer_idx in self.config.feature.layers:
            hidden_state = outputs.hidden_states[layer_idx]
            cls_token, patch_tokens = self._split_transformers_tokens(hidden_state, images)

            if self.config.feature.patch_features:
                features[f'patch_features_layer_{layer_idx}'] = patch_tokens.cpu()

            if self.config.feature.use_cls_token:
                global_features = cls_token
            else:
                global_features = patch_tokens.mean(dim=1)

            if self.config.feature.normalize:
                global_features = F.normalize(global_features, p=2, dim=1)

            global_key = f'global_features_layer_{layer_idx}'
            features[global_key] = global_features.cpu()
            if layer_idx == -1:
                features['global_features'] = global_features.cpu()

        return features
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> dict:
        """Extract features from a batch of images"""
        images = images.to(self.device)
        batch_size = images.shape[0]
        
        features = {}
        
        # Check if model has get_intermediate_layers method (torch.hub version)
        has_intermediate_layers = hasattr(self.model, 'get_intermediate_layers')
        
        if self.config.feature.patch_features and has_intermediate_layers:
            # Extract patch features from intermediate layers (torch.hub version)
            layer_outputs = self.model.get_intermediate_layers(
                images,
                n=self.config.feature.layers,
                reshape=True,
                norm=self.config.feature.normalize
            )
            
            # Process each layer output
            for i, layer_out in enumerate(layer_outputs):
                layer_idx = self.config.feature.layers[i] if i < len(self.config.feature.layers) else self.config.feature.layers[-1]
                
                # layer_out shape: [batch_size, embed_dim, h_patches, w_patches]
                patch_features = layer_out.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
                features[f'patch_features_layer_{layer_idx}'] = patch_features.cpu()
                
                if self.config.feature.use_cls_token:
                    # Global feature as mean of all patches
                    global_features = patch_features.mean(dim=1)  # [batch_size, embed_dim]
                    features[f'global_features_layer_{layer_idx}'] = global_features.cpu()
        
        else:
            # Extract global features only (works for both torch.hub and transformers)
            if hasattr(self.model, 'forward_features'):
                with torch.autocast(device_type=self.device.type, dtype=self.model_context['autocast_dtype']):
                    # torch.hub version
                    global_features = self.model.forward_features(images)
                    if hasattr(global_features, 'shape') and len(global_features.shape) > 2:
                        # If it returns patch features, take the mean
                        global_features = global_features.mean(dim=1)

                if self.config.feature.normalize:
                    global_features = F.normalize(global_features, p=2, dim=1)

                features['global_features'] = global_features.cpu()
            else:
                features.update(self._extract_transformers_features(images))
        
        return features
    
    def process_single_image(self, image_path: Union[str, Path]) -> dict:
        """Process a single image and extract features"""
        image_tensor = self.image_processor.process_image(image_path).unsqueeze(0)
        features = self.extract_features(image_tensor)
        
        # Remove batch dimension
        for key, value in features.items():
            features[key] = value.squeeze(0)
        
        return features

    @staticmethod
    def _is_feature_input_image(path: Path, root: Path) -> bool:
        """Return whether an image path should be treated as RGB input."""
        relative_parts = path.relative_to(root).parts
        parent_parts = {part.lower() for part in relative_parts[:-1]}
        if parent_parts & {"mask", "masks"}:
            return False

        stem = path.stem.lower()
        if stem == "mask" or stem.endswith("_mask") or "mask_" in stem or "_mask_" in stem:
            return False

        return True
    
    def process_directory(self, input_dir: Union[str, Path]) -> dict:
        """Process all images in a directory"""
        input_path = Path(input_dir)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            p for p in input_path.rglob('*')
            if p.suffix.lower() in image_extensions and self._is_feature_input_image(p, input_path)
        ]
        
        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")
        
        logger.info(f"Found {len(image_paths)} images in {input_dir}")
        
        all_features = {}
        
        # Process in batches
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            batch_images = self.image_processor.process_batch(batch_paths)
            
            batch_features = self.extract_features(batch_images)
            
            # Store features with image names
            for j, image_path in enumerate(batch_paths):
                image_name = image_path.stem
                for key, features in batch_features.items():
                    if key not in all_features:
                        all_features[key] = {}
                    all_features[key][image_name] = features[j]
            
            logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(image_paths) + self.config.batch_size - 1)//self.config.batch_size}")
        
        return all_features
    
    def save_features(self, features: dict, output_path: Union[str, Path]):
        """Save extracted features to disk"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.save_format == 'npz':
            # Save as numpy arrays
            for feature_type, feature_data in features.items():
                if isinstance(feature_data, dict):
                    # Multiple images
                    save_path = output_path / f"{feature_type}.npz"
                    np.savez_compressed(save_path, **{k: v.numpy() for k, v in feature_data.items()})
                else:
                    # Single image
                    save_path = output_path / f"{feature_type}.npy"
                    np.save(save_path, feature_data.numpy())
                logger.info(f"Saved {feature_type} to {save_path}")
        
        elif self.config.save_format == 'pt':
            # Save as PyTorch tensors
            for feature_type, feature_data in features.items():
                save_path = output_path / f"{feature_type}.pt"
                torch.save(feature_data, save_path)
                logger.info(f"Saved {feature_type} to {save_path}")
        
        else:
            raise ValueError(f"Unsupported save format: {self.config.save_format}")


def build_config(cfg: DictConfig) -> ExtractionConfig:
    """Build runtime dataclasses from a Hydra config."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Convert nested dict to dataclass
    model_config = ModelConfig(**cfg_dict['model'])
    image_config = ImageConfig(**cfg_dict['image'])
    feature_config = FeatureConfig(**cfg_dict['feature'])
    visualization_config = VisualizationConfig(**cfg_dict.get('visualization', {}))
    
    # Merge extraction config
    extraction_dict = cfg_dict['extraction'].copy()
    extraction_dict.update({
        'model': model_config,
        'image': image_config, 
        'feature': feature_config,
        'visualization': visualization_config,
    })
    
    return ExtractionConfig(**extraction_dict)


def resolve_visualization_output_size(
    output_size: Union[List[int], str],
    input_path: Union[str, Path],
) -> Tuple[int, int]:
    """Resolve visualization output size as (height, width)."""
    if isinstance(output_size, str):
        if output_size not in {"input", "original"}:
            raise ValueError("visualization.output_size must be [height, width], 'input', or 'original'")

        input_path = Path(input_path)
        if not input_path.is_file():
            raise ValueError("visualization.output_size='input' is only supported for single-image inputs")

        with Image.open(input_path) as image:
            width, height = image.size
        return height, width

    if len(output_size) != 2:
        raise ValueError("visualization.output_size must contain exactly two values: [height, width]")
    return int(output_size[0]), int(output_size[1])


def run_visualizations(
    feature_dir: Union[str, Path],
    config: VisualizationConfig,
    input_path: Union[str, Path],
) -> None:
    """Run configured visualizations for saved features."""
    if not config.enabled:
        return

    from visualize_features import FeatureVisualizer

    feature_dir = Path(feature_dir)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    visualizer = FeatureVisualizer(feature_dir)
    visualizer.print_summary()

    output_size = resolve_visualization_output_size(config.output_size, input_path)
    if config.patch_features:
        for feature_key in sorted(k for k in visualizer.features if k.startswith("patch_features")):
            save_path = save_dir / f"{feature_key}.png"
            visualizer.visualize_patch_features(
                feature_key=feature_key,
                method=config.method,
                output_size=output_size,
                save_path=save_path,
                save_individual=config.save_individual,
            )

    if config.global_features:
        for feature_key in sorted(k for k in visualizer.features if k.startswith("global_features")):
            save_path = save_dir / f"{feature_key}.png"
            visualizer.visualize_global_features(
                feature_key=feature_key,
                method=config.method,
                save_path=save_path,
            )

    if config.compare:
        save_path = save_dir / "feature_comparison.png"
        visualizer.compare_features(
            feature_keys=list(visualizer.features.keys()),
            save_path=save_path,
        )


@hydra.main(version_base=None, config_path="../configs/demo", config_name="default")
def main(cfg: DictConfig):
    """Main feature extraction pipeline"""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        config = build_config(cfg)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
        print("CUDA not available, using CPU")
    
    try:
        logger.info("Starting feature extraction...")
        logger.info(f"Scene: {cfg.get('scene', {}).get('id', 'default_scene')}")
        logger.info(f"Experiment: {cfg.get('exp', {}).get('name', 'default')}")
        logger.info(f"Input: {config.input_path}")
        logger.info(f"Output: {config.output_path}")
        logger.info(f"Model: {config.model.dino_hub or config.model.local_model_path or config.model.config_file}")
        
        # Initialize feature extractor
        extractor = FeatureExtractor(config)
        
        # Check if input is file or directory
        input_path = Path(config.input_path)
        
        if input_path.is_file():
            # Process single image
            logger.info("Processing single image...")
            features = extractor.process_single_image(input_path)
            output_name = input_path.stem
            output_path = Path(config.output_path) / output_name
            
        elif input_path.is_dir():
            # Process directory
            logger.info("Processing directory...")
            features = extractor.process_directory(input_path)
            output_path = Path(config.output_path)
            
        else:
            raise ValueError(f"Input path does not exist: {config.input_path}")
        
        # Save features
        extractor.save_features(features, output_path)
        run_visualizations(output_path, config.visualization, input_path)
        logger.info("Feature extraction completed!")
        
        # Print summary
        for feature_type in features.keys():
            if isinstance(features[feature_type], dict):
                logger.info(f"Extracted {feature_type} for {len(features[feature_type])} images")
            else:
                logger.info(f"Extracted {feature_type} with shape {features[feature_type].shape}")
                
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

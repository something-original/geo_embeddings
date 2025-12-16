"""
EfficientNet module for raster image embeddings.

This module provides EfficientNet B7 model and utilities for generating
embeddings from raster images (maps, satellite imagery, etc.).

Main components:
- EfficientNetEmbedder: The model class
- RasterEmbedder: High-level interface for embedding generation
- get_default_transforms: Image preprocessing transforms
- get_embedding: Convenience function for single image embedding
"""

from .model import EfficientNetEmbedder
from .transforms import get_default_transforms, preprocess_image, preprocess_image_pil
from .embedder import RasterEmbedder, get_embedding

__all__ = [
    'EfficientNetEmbedder',
    'RasterEmbedder',
    'get_embedding',
    'get_default_transforms',
    'preprocess_image',
    'preprocess_image_pil',
]

__version__ = '0.1.0'

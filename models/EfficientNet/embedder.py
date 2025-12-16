"""
Embedding generation utilities for EfficientNet.

This module provides high-level functions for generating embeddings
from raster images using EfficientNet B7 model.
"""

import os
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
from tqdm import tqdm
import torch

from .model import EfficientNetEmbedder
from .transforms import preprocess_image, get_default_transforms


class RasterEmbedder:
    """
    High-level interface for generating embeddings from raster images.
    """

    def __init__(self, model: Optional[EfficientNetEmbedder] = None, device: str = None):
        """
        Initialize the raster embedder.

        Args:
            model: Optional EfficientNetEmbedder instance. If None, creates a new one.
            device: Device to run the model on ('cpu', 'cuda', etc.). 
                   If None, uses 'cuda' if available, else 'cpu'.
        """
        if model is None:
            model = EfficientNetEmbedder(pretrained=True)

        self.model = model

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_default_transforms()

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding vector as numpy array of shape (2560,)
        """
        img_tensor = preprocess_image(image_path, self.transform)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()

        return embedding

    def embed_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing

        Returns:
            Embeddings array of shape (num_images, 2560)
        """
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    img_tensor = preprocess_image(path, self.transform)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    # Use zero tensor as fallback
                    batch_tensors.append(torch.zeros(1, 3, 224, 224))

            if batch_tensors:
                batch = torch.cat(batch_tensors, dim=0).to(self.device)

                with torch.no_grad():
                    batch_embeddings = self.model(batch).cpu().numpy()
                    embeddings.append(batch_embeddings)

        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])

    def embed_directory(
        self, 
        directory: Union[str, Path], 
        output_dir: Union[str, Path] = None,
        pattern: str = "*.png",
        batch_size: int = 32,
        save_embeddings: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all images in a directory.

        Args:
            directory: Directory containing images
            output_dir: Directory to save embeddings. If None, embeddings are not saved.
            pattern: File pattern to match (e.g., "*.png", "*.jpg")
            batch_size: Batch size for processing
            save_embeddings: Whether to save embeddings to .npy files

        Returns:
            Dictionary mapping image names (without extension) to embeddings
        """
        directory = Path(directory)
        image_paths = list(directory.glob(pattern))

        if not image_paths:
            print(f"No images found in {directory} matching pattern {pattern}")
            return {}

        print(f"Found {len(image_paths)} images")

        # Generate embeddings
        embeddings_array = self.embed_images_batch(
            [str(p) for p in image_paths], 
            batch_size=batch_size
        )

        # Create dictionary mapping
        embeddings_dict = {}
        for i, img_path in enumerate(image_paths):
            # Extract image ID from filename (e.g., "map_123.png" -> "123")
            img_name = img_path.stem
            # Try to extract number from filename (common pattern: map_123.png)
            parts = img_name.split("_")
            if len(parts) > 1:
                img_id = parts[-1]
            else:
                img_id = img_name

            embedding = embeddings_array[i]
            embeddings_dict[img_id] = embedding

            # Save embedding if requested
            if save_embeddings and output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{img_id}.npy"
                np.save(output_path, embedding)

        return embeddings_dict


def get_embedding(image_path: str, model: Optional[EfficientNetEmbedder] = None) -> np.ndarray:
    """
    Convenience function to get embedding for a single image.

    Args:
        image_path: Path to the image file
        model: Optional EfficientNetEmbedder instance. If None, creates a new one.

    Returns:
        Embedding vector as numpy array of shape (2560,)
    """
    embedder = RasterEmbedder(model=model)
    return embedder.embed_image(image_path)

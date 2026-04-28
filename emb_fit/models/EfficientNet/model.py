"""
EfficientNet B7 model for raster image embeddings.

This module provides the EfficientNet B7 model configured for generating
embeddings from raster images (maps, satellite imagery, etc.).
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetEmbedder(nn.Module):
    """
    EfficientNet B7 model configured for embedding generation.

    The model uses pretrained EfficientNet B7 weights and removes the classifier
    to output feature embeddings of dimension 2560.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize EfficientNet B7 embedder.

        Args:
            pretrained: If True, loads pretrained ImageNet weights.
        """
        super(EfficientNetEmbedder, self).__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        self.model.classifier = nn.Identity()
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Embeddings tensor of shape (batch_size, 2560)
        """
        return self.model(x)

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the output embeddings."""
        return 2560

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a single or batch of images.

        Args:
            image_tensor: Image tensor of shape (batch_size, 3, 224, 224) or (3, 224, 224)

        Returns:
            Embedding tensor of shape (batch_size, 2560) or (2560,)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.forward(image_tensor)

        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)

        return embedding

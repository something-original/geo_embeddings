"""
Image preprocessing transforms for EfficientNet.

This module provides image preprocessing pipelines for preparing
raster images (maps, satellite imagery) for EfficientNet model.
"""

from torchvision import transforms
from PIL import Image
import torch


def get_default_transforms() -> transforms.Compose:
    """
    Get default preprocessing transforms for EfficientNet.

    The transforms include:
    - Resize to 200 (maintaining aspect ratio)
    - Center crop to 224x224
    - Convert to grayscale and replicate to 3 channels
    - Convert to tensor
    - Normalize with ImageNet statistics

    Returns:
        Composed transforms pipeline
    """
    return transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def preprocess_image(image_path: str, transform: transforms.Compose = None) -> torch.Tensor:
    """
    Preprocess a single image from file path.

    Args:
        image_path: Path to the image file
        transform: Optional transform pipeline. If None, uses default transforms.

    Returns:
        Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    if transform is None:
        transform = get_default_transforms()

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor


def preprocess_image_pil(image: Image.Image, transform: transforms.Compose = None) -> torch.Tensor:
    """
    Preprocess a PIL Image object.

    Args:
        image: PIL Image object
        transform: Optional transform pipeline. If None, uses default transforms.

    Returns:
        Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    if transform is None:
        transform = get_default_transforms()

    img_tensor = transform(image).unsqueeze(0)

    return img_tensor

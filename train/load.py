import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import json
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def get_dataloaders(
    base_dir, img_size=64, batch_size=64, num_workers=2, val_split=0.2, pin_memory=False
):
    """
    Load train, validation, and test DataLoaders.

    Args:
        base_dir (str): Path to dataset base directory containing 'train/' and 'test/' folders.
        img_size (int): Target size for resized images (img_size x img_size).
        batch_size (int): Batch size for all loaders.
        num_workers (int): Number of worker processes for data loading.
        val_split (float): Fraction of training set to use as validation.
        pin_memory (bool): Whether to use pinned memory for DataLoader (useful for CUDA, not necessary for CPU/Metal).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    mean_std_path = os.path.join(base_dir, "mean_std.json")

    if os.path.exists(mean_std_path):
        with open(mean_std_path, "r") as f:
            stats = json.load(f)
        mean = stats["mean"]
        std = stats["std"]
    else:
        basic_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

        temp_dataset = datasets.ImageFolder(root=train_dir, transform=basic_transform)
        loader = DataLoader(
            temp_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        mean = 0.0
        std = 0.0
        total_images_count = 0

        for images, _ in loader:
            b_size = images.size(0)
            images = images.view(b_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += b_size

        mean /= total_images_count
        std /= total_images_count

        mean = mean.tolist()
        std = std.tolist()

        with open(mean_std_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f)

    # Added heavier data augmentation to reduce model reliance on class distribution
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.2)), # random zoom/crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # increased rotation
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1  # stronger brightness/contrast/saturation jitter
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random Erasing (cutout style)
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    targets = [label for _, label in full_train_dataset]

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    # Change transform for val_dataset
    val_dataset.dataset.transform = val_test_transform

    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

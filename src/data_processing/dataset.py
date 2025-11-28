"""
PyTorch Dataset classes for CogniSense multimodal data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import pickle


class MultimodalAlzheimerDataset(Dataset):
    """
    Dataset for multimodal Alzheimer's detection

    Handles all 5 modalities with flexible loading
    """

    def __init__(
        self,
        data_dict,
        modalities=['eye', 'typing', 'drawing', 'gait'],
        transform=None
    ):
        """
        Args:
            data_dict: Dictionary with keys:
                - 'eye_tracking': list of numpy arrays (N, 100, 2)
                - 'typing': list of numpy arrays (N, 50, 5)
                - 'clock_drawing': list of PIL Images
                - 'gait': list of numpy arrays (N, 3, time_steps)
                - 'labels': list of labels (0=control, 1=AD)
            modalities: Which modalities to include
            transform: Optional transforms (for images)
        """
        self.data = data_dict
        self.modalities = modalities
        self.transform = transform
        self.labels = torch.FloatTensor(data_dict['labels'])

        # Validate data
        n_samples = len(self.labels)
        for mod in modalities:
            key = f'{mod}_tracking' if mod == 'eye' else f'clock_{mod}' if mod == 'drawing' else mod
            if key not in data_dict:
                raise ValueError(f"Missing data for modality: {mod} (key: {key})")
            if len(data_dict[key]) != n_samples:
                raise ValueError(f"Sample count mismatch for {mod}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a dictionary with all modality data for one sample
        """
        item = {'label': self.labels[idx]}

        # Eye tracking
        if 'eye' in self.modalities:
            eye_data = self.data['eye_tracking'][idx]
            item['eye'] = torch.FloatTensor(eye_data)

        # Typing
        if 'typing' in self.modalities:
            typing_data = self.data['typing'][idx]
            item['typing'] = torch.FloatTensor(typing_data)

        # Clock drawing
        if 'drawing' in self.modalities:
            clock_img = self.data['clock_drawing'][idx]
            if self.transform:
                clock_img = self.transform(clock_img)
            item['drawing'] = clock_img

        # Gait
        if 'gait' in self.modalities:
            gait_data = self.data['gait'][idx]
            item['gait'] = torch.FloatTensor(gait_data)

        return item


class SingleModalityDataset(Dataset):
    """
    Dataset for single modality training
    """

    def __init__(self, data, labels, modality='eye', transform=None):
        """
        Args:
            data: List of data samples
            labels: List of labels (0=control, 1=AD)
            modality: Which modality this is
            transform: Optional transforms
        """
        self.data = data
        self.labels = torch.FloatTensor(labels)
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Convert to tensor based on modality
        if self.modality == 'drawing':
            if self.transform:
                sample = self.transform(sample)
        else:
            sample = torch.FloatTensor(sample)

        return sample, label


def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets

    Args:
        dataset: PyTorch Dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from torch.utils.data import random_split

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def collate_multimodal(batch):
    """
    Custom collate function for multimodal data

    Handles variable-length sequences and missing modalities
    """
    # Separate labels
    labels = torch.stack([item['label'] for item in batch])

    # Collect each modality
    batch_dict = {'labels': labels}

    # Check which modalities are present
    sample_keys = batch[0].keys()

    if 'eye' in sample_keys:
        batch_dict['eye'] = torch.stack([item['eye'] for item in batch])

    if 'typing' in sample_keys:
        batch_dict['typing'] = torch.stack([item['typing'] for item in batch])

    if 'drawing' in sample_keys:
        batch_dict['drawing'] = torch.stack([item['drawing'] for item in batch])

    if 'gait' in sample_keys:
        batch_dict['gait'] = torch.stack([item['gait'] for item in batch])

    return batch_dict


if __name__ == "__main__":
    # Test dataset creation
    from synthetic_data_generator import generate_synthetic_dataset

    print("Testing MultimodalAlzheimerDataset...")

    # Generate synthetic data
    data = generate_synthetic_dataset(num_samples=50, ad_ratio=0.5)

    # Create dataset
    dataset = MultimodalAlzheimerDataset(data)
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Test __getitem__
    item = dataset[0]
    print(f"✓ Sample 0:")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")

    # Test data splits
    train, val, test = create_data_splits(dataset)
    print(f"✓ Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(train, batch_size=4, collate_fn=collate_multimodal)
    batch = next(iter(loader))
    print(f"✓ Batch loaded:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\n✅ Dataset tests passed!")

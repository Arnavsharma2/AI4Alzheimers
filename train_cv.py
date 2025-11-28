#!/usr/bin/env python3
"""
Cross-validation training script for CogniSense models.
Implements k-fold cross-validation to ensure robust performance estimates.

This demonstrates experimental rigor expected in a month-long project.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.models.eye_model import EyeTrackingModel
from src.models.typing_model import TypingModel
from src.models.drawing_model import ClockDrawingModel
from src.models.gait_model import GaitModel
from src.fusion.fusion_model import MultimodalFusionModel
from src.data_processing.dataset import MultimodalAlzheimerDataset, custom_collate_fn
from src.data_processing.synthetic_data_generator import generate_synthetic_dataset
from src.utils.training_utils import compute_metrics, EarlyStopping, train_epoch, evaluate


def train_single_fold(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=30,
    patience=5,
    fold_num=1
):
    """
    Train a model on a single fold.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        fold_num: Fold number for logging

    Returns:
        dict: Training history and best metrics
    """
    print(f"\n{'='*60}")
    print(f"  Training Fold {fold_num}")
    print(f"{'='*60}\n")

    model.to(device)
    early_stopping = EarlyStopping(patience=patience, mode='max')

    best_metrics = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    for epoch in range(num_epochs):
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")

        # Early stopping check
        if early_stopping(val_metrics['auc']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        # Track best metrics
        if best_metrics is None or val_metrics['auc'] > best_metrics['auc']:
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch + 1

    print(f"\nBest metrics at epoch {best_metrics['epoch']}:")
    print(f"  AUC: {best_metrics['auc']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")

    return {
        'history': history,
        'best_metrics': best_metrics
    }


def cross_validate_model(
    model_class,
    dataset,
    n_splits=5,
    batch_size=32,
    num_epochs=30,
    lr=0.001,
    weight_decay=0.01,
    patience=5,
    device=None,
    save_dir='models/cv_results'
):
    """
    Perform k-fold cross-validation on a model.

    Args:
        model_class: Model class to instantiate
        dataset: Complete dataset
        n_splits: Number of CV folds
        batch_size: Batch size
        num_epochs: Max epochs per fold
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        device: Device to train on
        save_dir: Directory to save results

    Returns:
        dict: Cross-validation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  {n_splits}-Fold Cross-Validation")
    print(f"  Model: {model_class.__name__}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Get labels for stratified splitting
    labels = np.array([dataset[i]['label'] for i in range(len(dataset))])

    # Initialize k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results
    fold_results = []
    all_metrics = {
        'accuracy': [],
        'auc': [],
        'f1': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'recall': []
    }

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Val samples: {len(val_idx)}")

        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        # Initialize model
        model = model_class()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Train fold
        fold_result = train_single_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            fold_num=fold
        )

        # Store results
        fold_results.append(fold_result)
        for metric, value in fold_result['best_metrics'].items():
            if metric in all_metrics:
                all_metrics[metric].append(value)

    # Compute statistics across folds
    cv_stats = {}
    for metric, values in all_metrics.items():
        cv_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")

    print(f"Model: {model_class.__name__}")
    print(f"Folds: {n_splits}\n")

    print("Performance across folds:")
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    for metric in ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity']:
        if metric in cv_stats:
            stats = cv_stats[metric]
            print(f"{metric:<15} {stats['mean']:.4f}    {stats['std']:.4f}    "
                  f"{stats['min']:.4f}    {stats['max']:.4f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    model_name = model_class.__name__
    results_path = os.path.join(save_dir, f'{model_name}_cv_results.json')

    # Prepare serializable results
    serializable_results = {
        'model': model_name,
        'n_splits': n_splits,
        'cv_stats': {
            metric: {k: v if k != 'values' else [float(x) for x in v]
                     for k, v in stats.items()}
            for metric, stats in cv_stats.items()
        },
        'fold_results': [
            {
                'best_metrics': {k: float(v) for k, v in fold['best_metrics'].items()}
            }
            for fold in fold_results
        ]
    }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    return {
        'cv_stats': cv_stats,
        'fold_results': fold_results,
        'serializable': serializable_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validation training for CogniSense models'
    )
    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        choices=['eye', 'typing', 'drawing', 'gait', 'fusion'],
        help='Modality to train'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Maximum epochs per fold'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/cv_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate dataset
    print(f"\nGenerating synthetic dataset...")
    if args.modality == 'fusion':
        modalities = ['eye', 'typing', 'drawing', 'gait']
    else:
        modalities = [args.modality]

    data_dict = generate_synthetic_dataset(
        num_samples=args.num_samples,
        modalities=modalities
    )

    dataset = MultimodalAlzheimerDataset(
        data_dict,
        modalities=modalities
    )

    print(f"✓ Generated {len(dataset)} samples")

    # Select model class
    model_map = {
        'eye': EyeTrackingModel,
        'typing': TypingModel,
        'drawing': ClockDrawingModel,
        'gait': GaitModel,
        'fusion': lambda: MultimodalFusionModel(fusion_type='attention')
    }

    model_class = model_map[args.modality]

    # Run cross-validation
    results = cross_validate_model(
        model_class=model_class,
        dataset=dataset,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        save_dir=args.output_dir
    )

    print(f"\n{'='*60}")
    print(f"  ✓ Cross-validation complete!")
    print(f"{'='*60}\n")

    # Print key findings
    print("Key findings:")
    auc_stats = results['cv_stats']['auc']
    print(f"  Mean AUC: {auc_stats['mean']:.4f} ± {auc_stats['std']:.4f}")

    acc_stats = results['cv_stats']['accuracy']
    print(f"  Mean Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")

    f1_stats = results['cv_stats']['f1']
    print(f"  Mean F1: {f1_stats['mean']:.4f} ± {f1_stats['std']:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Training script for CogniSense models

Supports training:
- Individual modality models
- Multimodal fusion model

Usage:
    python train.py --mode single --modality eye --epochs 30
    python train.py --mode fusion --epochs 50
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.models.eye_model import EyeTrackingClassifier
from src.models.typing_model import TypingClassifier
from src.models.drawing_model import ClockDrawingClassifier
from src.models.gait_model import GaitClassifier
from src.fusion.fusion_model import MultimodalFusionModel
from src.data_processing.synthetic_data_generator import generate_synthetic_dataset
from src.data_processing.dataset import (
    MultimodalAlzheimerDataset,
    SingleModalityDataset,
    create_data_splits,
    collate_multimodal
)
from src.utils.training_utils import (
    train_epoch,
    evaluate,
    compute_metrics,
    EarlyStopping,
    MetricsTracker,
    save_model
)
from src.utils.helpers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CogniSense models")

    parser.add_argument('--mode', type=str, choices=['single', 'fusion'], default='fusion',
                        help='Training mode: single modality or fusion')
    parser.add_argument('--modality', type=str, choices=['eye', 'typing', 'drawing', 'gait'],
                        help='Which modality to train (for single mode)')

    # Data args
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to real data (if None, uses synthetic)')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of synthetic samples to generate')

    # Training args
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')

    # Model args
    parser.add_argument('--freeze-encoders', action='store_true',
                        help='Freeze pretrained encoders')

    # Other args
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')

    return parser.parse_args()


def train_single_modality(args):
    """Train a single modality classifier"""
    print(f"\n{'='*60}")
    print(f"Training {args.modality.upper()} Model")
    print(f"{'='*60}\n")

    # Generate or load data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        # TODO: Implement real data loading
        raise NotImplementedError("Real data loading not yet implemented")
    else:
        print(f"Generating {args.num_samples} synthetic samples...")
        dataset_dict = generate_synthetic_dataset(args.num_samples, ad_ratio=0.5)

    # Create datasets
    modality_key = {
        'eye': 'eye_tracking',
        'typing': 'typing',
        'drawing': 'clock_drawing',
        'gait': 'gait'
    }[args.modality]

    dataset = SingleModalityDataset(
        data=dataset_dict[modality_key],
        labels=dataset_dict['labels'],
        modality=args.modality
    )

    # Split data
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset)
    print(f"Splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model_classes = {
        'eye': EyeTrackingClassifier,
        'typing': TypingClassifier,
        'drawing': ClockDrawingClassifier,
        'gait': GaitClassifier
    }

    model_config = {}
    if args.modality in ['drawing'] and args.freeze_encoders:
        model_config['model_config'] = {'freeze_encoder': True}

    model = model_classes[args.modality](model_config.get('model_config', {}))
    model = model.to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training setup
    early_stopping = EarlyStopping(patience=5, mode='min')
    metrics_tracker = MetricsTracker(save_dir=Path(args.save_dir) / args.modality)

    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_preds, train_labels, train_probs = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        train_metrics = compute_metrics(train_labels, train_preds, train_probs)

        # Validate
        val_loss, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, args.device
        )
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        # Track metrics
        metrics_tracker.update(epoch, train_loss, val_loss, train_metrics, val_metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_dir) / args.modality / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, optimizer, epoch, val_metrics, save_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}\n")

    # Load best model
    checkpoint = torch.load(Path(args.save_dir) / args.modality / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, args.device
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")

    # Save test metrics
    with open(Path(args.save_dir) / args.modality / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def train_fusion(args):
    """Train the multimodal fusion model"""
    print(f"\n{'='*60}")
    print(f"Training Multimodal Fusion Model")
    print(f"{'='*60}\n")

    # Generate or load data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        raise NotImplementedError("Real data loading not yet implemented")
    else:
        print(f"Generating {args.num_samples} synthetic samples...")
        dataset_dict = generate_synthetic_dataset(args.num_samples, ad_ratio=0.5)

    # Create dataset
    from transformers import ViTImageProcessor
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Transform for images
    def transform_image(img):
        processed = vit_processor(images=img, return_tensors="pt")
        return processed['pixel_values'].squeeze(0)

    dataset = MultimodalAlzheimerDataset(
        dataset_dict,
        modalities=['eye', 'typing', 'drawing', 'gait'],
        transform=transform_image
    )

    # Split data
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset)
    print(f"Splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           collate_fn=collate_multimodal)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            collate_fn=collate_multimodal)

    # Create model (without speech for now since we don't have real audio)
    model = MultimodalFusionModel(
        speech_config={'freeze_encoders': True},
        drawing_config={'freeze_encoder': args.freeze_encoders},
        fusion_type='attention'
    )
    model = model.to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training setup
    early_stopping = EarlyStopping(patience=7, mode='min')
    metrics_tracker = MetricsTracker(save_dir=Path(args.save_dir) / 'fusion')

    best_val_loss = float('inf')

    # Simplified training loop for fusion (no speech)
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Training epoch
        model.train()
        train_loss = 0
        train_preds_list = []
        train_labels_list = []
        train_probs_list = []

        from tqdm import tqdm
        for batch in tqdm(train_loader, desc="Training"):
            labels = batch['labels'].to(args.device)

            # Prepare inputs (no speech)
            inputs = {}
            if 'eye' in batch:
                inputs['eye_gaze'] = batch['eye'].to(args.device)
            if 'typing' in batch:
                inputs['typing_sequence'] = batch['typing'].to(args.device)
            if 'drawing' in batch:
                inputs['drawing_image'] = batch['drawing'].to(args.device)
            if 'gait' in batch:
                inputs['gait_sensor'] = batch['gait'].to(args.device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = outputs.squeeze().detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            train_probs_list.extend(probs)
            train_preds_list.extend(preds)
            train_labels_list.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_metrics = compute_metrics(train_labels_list, train_preds_list, train_probs_list)

        # Validation
        model.eval()
        val_loss = 0
        val_preds_list = []
        val_labels_list = []
        val_probs_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                labels = batch['labels'].to(args.device)

                inputs = {}
                if 'eye' in batch:
                    inputs['eye_gaze'] = batch['eye'].to(args.device)
                if 'typing' in batch:
                    inputs['typing_sequence'] = batch['typing'].to(args.device)
                if 'drawing' in batch:
                    inputs['drawing_image'] = batch['drawing'].to(args.device)
                if 'gait' in batch:
                    inputs['gait_sensor'] = batch['gait'].to(args.device)

                outputs = model(**inputs)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item()
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_probs_list.extend(probs)
                val_preds_list.extend(preds)
                val_labels_list.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_metrics = compute_metrics(val_labels_list, val_preds_list, val_probs_list)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        # Track metrics
        metrics_tracker.update(epoch, train_loss, val_loss, train_metrics, val_metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_dir) / 'fusion' / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, optimizer, epoch, val_metrics, save_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}\n")

    checkpoint = torch.load(Path(args.save_dir) / 'fusion' / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_loss = 0
    test_preds_list = []
    test_labels_list = []
    test_probs_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            labels = batch['labels'].to(args.device)

            inputs = {}
            if 'eye' in batch:
                inputs['eye_gaze'] = batch['eye'].to(args.device)
            if 'typing' in batch:
                inputs['typing_sequence'] = batch['typing'].to(args.device)
            if 'drawing' in batch:
                inputs['drawing_image'] = batch['drawing'].to(args.device)
            if 'gait' in batch:
                inputs['gait_sensor'] = batch['gait'].to(args.device)

            outputs = model(**inputs)
            loss = criterion(outputs.squeeze(), labels)

            test_loss += loss.item()
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            test_probs_list.extend(probs)
            test_preds_list.extend(preds)
            test_labels_list.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_metrics = compute_metrics(test_labels_list, test_preds_list, test_probs_list)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")

    # Save test metrics
    with open(Path(args.save_dir) / 'fusion' / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Print configuration
    print("\n" + "="*60)
    print("CogniSense Training")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.mode == 'single':
        if not args.modality:
            raise ValueError("Must specify --modality for single mode")
        print(f"Modality: {args.modality}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)

    # Train
    if args.mode == 'single':
        metrics = train_single_modality(args)
    else:
        metrics = train_fusion(args)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

    return metrics


if __name__ == "__main__":
    main()

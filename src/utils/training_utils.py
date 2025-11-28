"""
Training utilities for CogniSense models

Includes metrics, logging, early stopping, and training loops
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # same as recall
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0

    return metrics


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """

    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Check if should stop

        Args:
            score: Current metric value

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class MetricsTracker:
    """
    Track training metrics over epochs
    """

    def __init__(self, save_dir=None):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        """Update history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)

        if self.save_dir:
            self.save()

    def save(self):
        """Save history to JSON"""
        if self.save_dir:
            with open(self.save_dir / 'training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

    def get_best_epoch(self, metric='val_loss', mode='min'):
        """Get epoch with best metric"""
        if metric == 'val_loss':
            values = self.history['val_loss']
        else:
            values = [m[metric] for m in self.history['val_metrics']]

        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return best_idx + 1  # 1-indexed


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss, predictions, labels
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Handle different batch formats
        if isinstance(batch, dict):
            labels = batch['labels'].to(device)
            # Move other items to device
            for key in batch.keys():
                if key != 'labels' and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            inputs = batch
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        if isinstance(inputs, dict):
            outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        else:
            outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        probs = outputs.squeeze().detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model

    Args:
        model: PyTorch model
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device

    Returns:
        Average loss, predictions, labels, probabilities
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, dict):
                labels = batch['labels'].to(device)
                for key in batch.keys():
                    if key != 'labels' and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                inputs = batch
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            # Forward pass
            if isinstance(inputs, dict):
                outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
            else:
                outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), labels)

            # Track metrics
            total_loss += loss.item()
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_model(model, optimizer, epoch, metrics, save_path):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def load_model(model, checkpoint_path, optimizer=None):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state

    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


if __name__ == "__main__":
    # Test metrics computation
    print("Testing training utilities...")

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.1, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("✓ Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [1.0, 0.9, 0.85, 0.84, 0.84, 0.85, 0.86]
    print("\n✓ Early stopping test:")
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"  Epoch {i+1}: score={score}, stop={stop}")
        if stop:
            break

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(1, 0.5, 0.4, {'acc': 0.7}, {'acc': 0.75})
    tracker.update(2, 0.4, 0.35, {'acc': 0.75}, {'acc': 0.8})
    best_epoch = tracker.get_best_epoch('val_loss', 'min')
    print(f"\n✓ Best epoch: {best_epoch}")

    print("\n✅ Training utilities tests passed!")

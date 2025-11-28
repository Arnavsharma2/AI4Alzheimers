"""
Visualization utilities for CogniSense

Functions for creating publication-quality plots and figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
from pathlib import Path


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']


def plot_roc_curve(y_true, y_probs, labels=None, save_path=None):
    """
    Plot ROC curve(s)

    Args:
        y_true: True labels (single array or list of arrays)
        y_probs: Predicted probabilities (single array or list of arrays)
        labels: List of labels for each curve
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Handle single or multiple curves
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_probs = [y_probs]
        labels = [labels] if labels else ['Model']

    # Plot each ROC curve
    for i, (yt, yp, label) in enumerate(zip(y_true, y_probs, labels)):
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2.5,
                label=f'{label} (AUC = {roc_auc:.3f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, labels=['Control', 'AD'],
                          normalize=False, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize
        save_path: Path to save figure

    Returns:
        Figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})

    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_metrics', 'val_metrics'
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, history['val_loss'], 'r-o', label='Validation', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    ax.plot(epochs, train_acc, 'b-o', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, val_acc, 'r-o', label='Validation', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1, 0]
    train_auc = [m.get('auc', 0) for m in history['train_metrics']]
    val_auc = [m.get('auc', 0) for m in history['val_metrics']]
    ax.plot(epochs, train_auc, 'b-o', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, val_auc, 'r-o', label='Validation', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 1]
    train_f1 = [m.get('f1', 0) for m in history['train_metrics']]
    val_f1 = [m.get('f1', 0) for m in history['val_metrics']]
    ax.plot(epochs, train_f1, 'b-o', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, val_f1, 'r-o', label='Validation', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation F1', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")

    return fig


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of metrics across models

    Args:
        metrics_dict: Dictionary {model_name: {metric: value}}
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T

    # Select key metrics
    metrics_to_plot = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    df_plot = df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar plot
    df_plot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(title='Metric', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {save_path}")

    return fig


def plot_attention_heatmap(attention_weights, modality_names=None, sample_ids=None,
                           save_path=None):
    """
    Plot attention weights as heatmap

    Args:
        attention_weights: Array of shape (n_samples, n_modalities)
        modality_names: List of modality names
        sample_ids: List of sample IDs/labels
        save_path: Path to save figure

    Returns:
        Figure object
    """
    if modality_names is None:
        modality_names = [f'Modality {i+1}' for i in range(attention_weights.shape[1])]

    if sample_ids is None:
        sample_ids = [f'Sample {i+1}' for i in range(attention_weights.shape[0])]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sample_ids) * 0.4)))

    sns.heatmap(attention_weights, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=modality_names, yticklabels=sample_ids,
                cbar_kws={'label': 'Attention Weight'},
                ax=ax, annot_kws={'size': 10})

    ax.set_xlabel('Modality', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sample', fontsize=13, fontweight='bold')
    ax.set_title('Attention Weights Heatmap', fontsize=15, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention heatmap saved to {save_path}")

    return fig


def plot_ablation_study(results, save_path=None):
    """
    Plot ablation study results

    Args:
        results: Dictionary {n_modalities: {'auc': value, 'accuracy': value}}
        save_path: Path to save figure

    Returns:
        Figure object
    """
    n_mods = sorted(results.keys())
    aucs = [results[n]['auc'] for n in n_mods]
    accs = [results[n]['accuracy'] for n in n_mods]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(n_mods))
    width = 0.35

    ax.bar(x - width/2, aucs, width, label='AUC', color='#4ECDC4',
           edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, accs, width, label='Accuracy', color='#FF6B6B',
           edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Number of Modalities', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Performance vs. Number of Modalities',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(n_mods)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (auc_val, acc_val) in enumerate(zip(aucs, accs)):
        ax.text(i - width/2, auc_val + 0.02, f'{auc_val:.3f}',
                ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, acc_val + 0.02, f'{acc_val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Ablation study plot saved to {save_path}")

    return fig


def create_results_summary_table(metrics_dict, save_path=None):
    """
    Create a formatted table of results

    Args:
        metrics_dict: Dictionary {model_name: {metric: value}}
        save_path: Path to save table (PNG)

    Returns:
        Figure object
    """
    df = pd.DataFrame(metrics_dict).T

    # Round values
    df = df.round(4)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(len(df)):
        table[(i+1, -1)].set_facecolor('#F0F0F0')
        table[(i+1, -1)].set_text_props(weight='bold')

    # Highlight best values
    for col_idx, col in enumerate(df.columns):
        best_idx = df[col].idxmax()
        row_idx = df.index.get_loc(best_idx) + 1
        table[(row_idx, col_idx)].set_facecolor('#90EE90')

    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results table saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...\n")

    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_probs = np.random.rand(100)
    y_pred = (y_probs > 0.5).astype(int)

    # Test ROC curve
    print("1. Testing ROC curve...")
    fig = plot_roc_curve(y_true, y_probs, labels='Test Model')
    plt.close()
    print("   ✓ ROC curve created\n")

    # Test confusion matrix
    print("2. Testing confusion matrix...")
    fig = plot_confusion_matrix(y_true, y_pred)
    plt.close()
    print("   ✓ Confusion matrix created\n")

    # Test training curves
    print("3. Testing training curves...")
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28],
        'val_loss': [0.52, 0.42, 0.38, 0.35, 0.33],
        'train_metrics': [
            {'accuracy': 0.7, 'auc': 0.75, 'f1': 0.68},
            {'accuracy': 0.75, 'auc': 0.78, 'f1': 0.73},
            {'accuracy': 0.78, 'auc': 0.82, 'f1': 0.76},
            {'accuracy': 0.80, 'auc': 0.84, 'f1': 0.78},
            {'accuracy': 0.82, 'auc': 0.86, 'f1': 0.80}
        ],
        'val_metrics': [
            {'accuracy': 0.68, 'auc': 0.73, 'f1': 0.66},
            {'accuracy': 0.72, 'auc': 0.76, 'f1': 0.70},
            {'accuracy': 0.75, 'auc': 0.80, 'f1': 0.73},
            {'accuracy': 0.77, 'auc': 0.82, 'f1': 0.75},
            {'accuracy': 0.78, 'auc': 0.84, 'f1': 0.76}
        ]
    }
    fig = plot_training_curves(history)
    plt.close()
    print("   ✓ Training curves created\n")

    # Test metrics comparison
    print("4. Testing metrics comparison...")
    metrics = {
        'Speech': {'accuracy': 0.74, 'auc': 0.78, 'sensitivity': 0.76, 'specificity': 0.72, 'f1': 0.74},
        'Eye': {'accuracy': 0.69, 'auc': 0.72, 'sensitivity': 0.70, 'specificity': 0.68, 'f1': 0.69},
        'Typing': {'accuracy': 0.67, 'auc': 0.70, 'sensitivity': 0.69, 'specificity': 0.65, 'f1': 0.67},
        'Drawing': {'accuracy': 0.79, 'auc': 0.82, 'sensitivity': 0.81, 'specificity': 0.77, 'f1': 0.79},
        'Gait': {'accuracy': 0.71, 'auc': 0.75, 'sensitivity': 0.73, 'specificity': 0.69, 'f1': 0.71},
        'Fusion': {'accuracy': 0.85, 'auc': 0.89, 'sensitivity': 0.87, 'specificity': 0.83, 'f1': 0.85}
    }
    fig = plot_metrics_comparison(metrics)
    plt.close()
    print("   ✓ Metrics comparison created\n")

    # Test attention heatmap
    print("5. Testing attention heatmap...")
    attention = np.random.rand(5, 5)
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize
    fig = plot_attention_heatmap(
        attention,
        modality_names=['Speech', 'Eye', 'Typing', 'Drawing', 'Gait'],
        sample_ids=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']
    )
    plt.close()
    print("   ✓ Attention heatmap created\n")

    # Test ablation study
    print("6. Testing ablation study...")
    ablation_results = {
        1: {'auc': 0.75, 'accuracy': 0.71},
        2: {'auc': 0.81, 'accuracy': 0.77},
        3: {'auc': 0.85, 'accuracy': 0.81},
        4: {'auc': 0.88, 'accuracy': 0.84},
        5: {'auc': 0.89, 'accuracy': 0.85}
    }
    fig = plot_ablation_study(ablation_results)
    plt.close()
    print("   ✓ Ablation study plot created\n")

    # Test summary table
    print("7. Testing summary table...")
    fig = create_results_summary_table(metrics)
    plt.close()
    print("   ✓ Summary table created\n")

    print("="*60)
    print("✅ All visualization tests passed!")
    print("="*60)

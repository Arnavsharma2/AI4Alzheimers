"""
Error analysis and model interpretability utilities.
Provides tools to understand model behavior and failure modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch


def analyze_errors(y_true, y_pred, y_prob, feature_names=None, threshold=0.5):
    """
    Analyze model errors and identify patterns.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        feature_names: Optional list of feature names
        threshold: Classification threshold

    Returns:
        dict: Error analysis results
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Identify error types
    true_positives = (y_true == 1) & (y_pred == 1)
    true_negatives = (y_true == 0) & (y_pred == 0)
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)

    # Calculate confidence for each prediction
    confidence = np.abs(y_prob - 0.5) * 2  # 0 = uncertain, 1 = confident

    # Analyze false positives
    fp_indices = np.where(false_positives)[0]
    fp_confidences = confidence[fp_indices]

    # Analyze false negatives
    fn_indices = np.where(false_negatives)[0]
    fn_confidences = confidence[fn_indices]

    # High confidence errors are particularly problematic
    high_conf_threshold = 0.7
    high_conf_fp = fp_indices[fp_confidences > high_conf_threshold]
    high_conf_fn = fn_indices[fn_confidences > high_conf_threshold]

    results = {
        'total_samples': len(y_true),
        'correct': int((y_true == y_pred).sum()),
        'incorrect': int((y_true != y_pred).sum()),
        'false_positives': {
            'count': int(false_positives.sum()),
            'indices': fp_indices.tolist(),
            'avg_confidence': float(fp_confidences.mean()) if len(fp_confidences) > 0 else 0.0,
            'high_confidence_count': len(high_conf_fp),
            'high_confidence_indices': high_conf_fp.tolist()
        },
        'false_negatives': {
            'count': int(false_negatives.sum()),
            'indices': fn_indices.tolist(),
            'avg_confidence': float(fn_confidences.mean()) if len(fn_confidences) > 0 else 0.0,
            'high_confidence_count': len(high_conf_fn),
            'high_confidence_indices': high_conf_fn.tolist()
        },
        'error_rate_by_confidence': analyze_error_rate_by_confidence(
            y_true, y_pred, confidence
        )
    }

    return results


def analyze_error_rate_by_confidence(y_true, y_pred, confidence, n_bins=5):
    """
    Analyze how error rate varies with prediction confidence.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence: Prediction confidence scores
        n_bins: Number of confidence bins

    Returns:
        dict: Error rates for each confidence bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_results = []

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (confidence >= bins[i]) & (confidence <= bins[i+1])

        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            error_rate = (bin_true != bin_pred).mean()
        else:
            error_rate = 0.0

        bin_results.append({
            'confidence_range': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
            'count': int(mask.sum()),
            'error_rate': float(error_rate)
        })

    return bin_results


def plot_error_analysis(error_results, save_path=None):
    """
    Visualize error analysis results.

    Args:
        error_results: dict from analyze_errors()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error distribution
    ax = axes[0]
    categories = ['False Positives', 'False Negatives']
    counts = [
        error_results['false_positives']['count'],
        error_results['false_negatives']['count']
    ]
    colors = ['#ff6b6b', '#ffa06b']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')

    # Plot 2: Error rate by confidence
    ax = axes[1]
    conf_bins = error_results['error_rate_by_confidence']
    bin_labels = [b['confidence_range'] for b in conf_bins]
    error_rates = [b['error_rate'] * 100 for b in conf_bins]

    bars = ax.bar(range(len(bin_labels)), error_rates, color='#6ba3ff',
                   alpha=0.7, edgecolor='black')
    ax.set_xlabel('Confidence Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Error Rate by Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, rate in zip(bars, error_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error analysis plot saved to {save_path}")

    return fig


def compute_uncertainty(model, data_loader, device, n_iterations=10):
    """
    Compute prediction uncertainty using Monte Carlo dropout.

    Args:
        model: PyTorch model (must have dropout layers)
        data_loader: Data loader
        device: Device
        n_iterations: Number of forward passes

    Returns:
        tuple: (mean predictions, uncertainty scores)
    """
    model.train()  # Enable dropout

    all_predictions = []

    with torch.no_grad():
        for _ in range(n_iterations):
            predictions = []

            for batch in data_loader:
                # Handle different batch structures
                if isinstance(batch, dict):
                    # Get first available modality
                    for key in ['eye', 'typing', 'drawing', 'gait']:
                        if key in batch:
                            inputs = batch[key]
                            break
                else:
                    inputs = batch[0]

                # Move to device and run inference
                if isinstance(inputs, list):
                    outputs = []
                    for inp in inputs:
                        inp_tensor = torch.FloatTensor(inp).unsqueeze(0).to(device)
                        out = model(inp_tensor)
                        outputs.append(torch.sigmoid(out).cpu().numpy())
                    predictions.extend([o.item() for o in outputs])
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    predictions.extend(torch.sigmoid(outputs).cpu().numpy().tolist())

            all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)  # Shape: (n_iterations, n_samples)

    # Compute statistics
    mean_pred = all_predictions.mean(axis=0)
    std_pred = all_predictions.std(axis=0)

    # Uncertainty score (higher = more uncertain)
    uncertainty = std_pred

    return mean_pred, uncertainty


def analyze_attention_patterns(attention_weights, modality_names, labels=None):
    """
    Analyze attention patterns to understand modality importance.

    Args:
        attention_weights: numpy array (n_samples, n_modalities)
        modality_names: list of modality names
        labels: Optional labels for stratified analysis

    Returns:
        dict: Attention pattern analysis
    """
    attention_weights = np.array(attention_weights)

    # Overall statistics
    mean_attention = attention_weights.mean(axis=0)
    std_attention = attention_weights.std(axis=0)

    results = {
        'overall': {
            'mean': {modality_names[i]: float(mean_attention[i])
                     for i in range(len(modality_names))},
            'std': {modality_names[i]: float(std_attention[i])
                    for i in range(len(modality_names))}
        }
    }

    # Stratified analysis by label
    if labels is not None:
        labels = np.array(labels)
        for label_val in np.unique(labels):
            mask = labels == label_val
            label_attention = attention_weights[mask]
            label_mean = label_attention.mean(axis=0)

            label_name = 'AD' if label_val == 1 else 'Control'
            results[label_name] = {
                modality_names[i]: float(label_mean[i])
                for i in range(len(modality_names))
            }

    # Find most and least important modalities
    most_important_idx = mean_attention.argmax()
    least_important_idx = mean_attention.argmin()

    results['most_important'] = modality_names[most_important_idx]
    results['least_important'] = modality_names[least_important_idx]

    return results


def plot_attention_patterns(attention_results, save_path=None):
    """
    Visualize attention pattern analysis.

    Args:
        attention_results: dict from analyze_attention_patterns()
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    modalities = list(attention_results['overall']['mean'].keys())
    overall_mean = [attention_results['overall']['mean'][m] for m in modalities]
    overall_std = [attention_results['overall']['std'][m] for m in modalities]

    x = np.arange(len(modalities))
    width = 0.35

    # Plot overall
    bars1 = ax.bar(x - width/2, overall_mean, width,
                   yerr=overall_std, label='Overall',
                   color='#6ba3ff', alpha=0.7, capsize=5,
                   edgecolor='black')

    # Plot by class if available
    if 'AD' in attention_results:
        ad_values = [attention_results['AD'][m] for m in modalities]
        control_values = [attention_results['Control'][m] for m in modalities]

        bars2 = ax.bar(x - width*1.5, control_values, width,
                      label='Control', color='#6bff7a', alpha=0.7,
                      edgecolor='black')
        bars3 = ax.bar(x - width/2, ad_values, width,
                      label='AD', color='#ff6b6b', alpha=0.7,
                      edgecolor='black')

    ax.set_xlabel('Modality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Attention Patterns Across Modalities', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention pattern plot saved to {save_path}")

    return fig


def generate_error_report(y_true, y_pred, y_prob, output_path):
    """
    Generate a comprehensive error analysis report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        output_path: Path to save report
    """
    # Perform analysis
    error_results = analyze_errors(y_true, y_pred, y_prob)

    # Generate report
    report = []
    report.append("="*70)
    report.append("  MODEL ERROR ANALYSIS REPORT")
    report.append("="*70)
    report.append("")

    report.append(f"Total samples: {error_results['total_samples']}")
    report.append(f"Correct predictions: {error_results['correct']} "
                  f"({error_results['correct']/error_results['total_samples']*100:.1f}%)")
    report.append(f"Incorrect predictions: {error_results['incorrect']} "
                  f"({error_results['incorrect']/error_results['total_samples']*100:.1f}%)")
    report.append("")

    report.append("FALSE POSITIVES")
    report.append("-" * 70)
    fp = error_results['false_positives']
    report.append(f"Count: {fp['count']}")
    report.append(f"Average confidence: {fp['avg_confidence']:.3f}")
    report.append(f"High confidence errors: {fp['high_confidence_count']}")
    report.append("")

    report.append("FALSE NEGATIVES")
    report.append("-" * 70)
    fn = error_results['false_negatives']
    report.append(f"Count: {fn['count']}")
    report.append(f"Average confidence: {fn['avg_confidence']:.3f}")
    report.append(f"High confidence errors: {fn['high_confidence_count']}")
    report.append("")

    report.append("ERROR RATE BY CONFIDENCE")
    report.append("-" * 70)
    for bin_info in error_results['error_rate_by_confidence']:
        report.append(f"{bin_info['confidence_range']}: "
                     f"{bin_info['error_rate']*100:.1f}% "
                     f"(n={bin_info['count']})")
    report.append("")

    report.append("="*70)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Error analysis report saved to {output_path}")

    return '\n'.join(report)

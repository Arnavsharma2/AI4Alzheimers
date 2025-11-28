# Visualization Guide for CogniSense

Utilities for creating publication-quality plots and figures.

---

## Available Visualizations

### 1. ROC Curves

Plot Receiver Operating Characteristic curves to visualize model discrimination ability.

```python
from src.utils.visualization import plot_roc_curve

# Single model
plot_roc_curve(y_true, y_probs, labels='CogniSense', save_path='roc.png')

# Multiple models
plot_roc_curve(
    [y_true_1, y_true_2],
    [y_probs_1, y_probs_2],
    labels=['Model 1', 'Model 2'],
    save_path='roc_comparison.png'
)
```

### 2. Confusion Matrix

Visualize classification performance with confusion matrices.

```python
from src.utils.visualization import plot_confusion_matrix

# Raw counts
plot_confusion_matrix(y_true, y_pred, labels=['Control', 'AD'])

# Normalized
plot_confusion_matrix(y_true, y_pred, normalize=True, save_path='cm.png')
```

### 3. Training Curves

Monitor training progress with loss/accuracy curves.

```python
from src.utils.visualization import plot_training_curves
import json

# Load training history
with open('checkpoints/fusion/training_history.json') as f:
    history = json.load(f)

plot_training_curves(history, save_path='training_curves.png')
```

### 4. Metrics Comparison

Compare performance across models.

```python
from src.utils.visualization import plot_metrics_comparison

metrics = {
    'Speech': {'accuracy': 0.74, 'auc': 0.78, ...},
    'Eye': {'accuracy': 0.69, 'auc': 0.72, ...},
    'Fusion': {'accuracy': 0.85, 'auc': 0.89, ...}
}

plot_metrics_comparison(metrics, save_path='comparison.png')
```

### 5. Attention Heatmap

Visualize attention weights across samples and modalities.

```python
from src.utils.visualization import plot_attention_heatmap

attention_weights = model.get_attention_weights(data)  # (n_samples, n_modalities)

plot_attention_heatmap(
    attention_weights,
    modality_names=['Speech', 'Eye', 'Typing', 'Drawing', 'Gait'],
    sample_ids=['Patient 1', 'Patient 2', ...],
    save_path='attention.png'
)
```

### 6. Ablation Study

Show performance improvement with additional modalities.

```python
from src.utils.visualization import plot_ablation_study

results = {
    1: {'auc': 0.75, 'accuracy': 0.71},  # Best single modality
    2: {'auc': 0.81, 'accuracy': 0.77},  # Best 2 modalities
    3: {'auc': 0.85, 'accuracy': 0.81},
    4: {'auc': 0.88, 'accuracy': 0.84},
    5: {'auc': 0.89, 'accuracy': 0.85}   # All modalities
}

plot_ablation_study(results, save_path='ablation.png')
```

### 7. Results Summary Table

Create formatted table of all model results.

```python
from src.utils.visualization import create_results_summary_table

metrics = {
    'Speech': {'accuracy': 0.74, 'auc': 0.78, 'sensitivity': 0.76, ...},
    'Fusion': {'accuracy': 0.85, 'auc': 0.89, 'sensitivity': 0.87, ...}
}

create_results_summary_table(metrics, save_path='results_table.png')
```

---

## Complete Example: Generate All Plots

```python
import json
import numpy as np
from src.utils.visualization import *

# Load results
with open('checkpoints/fusion/training_history.json') as f:
    history = json.load(f)

with open('checkpoints/fusion/test_metrics.json') as f:
    test_metrics = json.load(f)

# 1. Training curves
plot_training_curves(history, save_path='results/figures/training_curves.png')

# 2. ROC curve
y_true = np.load('results/y_true.npy')
y_probs = np.load('results/y_probs.npy')
plot_roc_curve(y_true, y_probs, labels='CogniSense Fusion',
               save_path='results/figures/roc_curve.png')

# 3. Confusion matrix
y_pred = (y_probs > 0.5).astype(int)
plot_confusion_matrix(y_true, y_pred, normalize=True,
                     save_path='results/figures/confusion_matrix.png')

# 4. Metrics comparison
all_metrics = {
    'Speech': {...},  # Load from checkpoints/speech/test_metrics.json
    'Eye': {...},
    'Typing': {...},
    'Drawing': {...},
    'Gait': {...},
    'Fusion': test_metrics
}
plot_metrics_comparison(all_metrics, save_path='results/figures/comparison.png')

# 5. Attention heatmap
attention_weights = np.load('results/attention_weights.npy')
plot_attention_heatmap(attention_weights,
                      modality_names=['Speech', 'Eye', 'Typing', 'Drawing', 'Gait'],
                      save_path='results/figures/attention.png')

# 6. Ablation study
ablation_results = {1: {...}, 2: {...}, 3: {...}, 4: {...}, 5: {...}}
plot_ablation_study(ablation_results, save_path='results/figures/ablation.png')

# 7. Summary table
create_results_summary_table(all_metrics, save_path='results/figures/summary_table.png')

print("✅ All visualizations generated!")
```

---

## Styling

All plots use consistent styling:
- **Color palette**: Seaborn 'husl' (7 distinct colors)
- **Grid**: Semi-transparent for readability
- **Fonts**: Bold titles and labels
- **DPI**: 300 for publication quality
- **Format**: PNG with tight bounding box

### Custom Colors

```python
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
```

---

## Tips

### For Papers/Reports

- Use `dpi=300` for high quality
- Save as PNG or PDF format
- Include clear legends and axis labels
- Normalize confusion matrices

### For Presentations

- Use larger font sizes (add `fontsize` parameter)
- Brighter colors for visibility
- Simplify plots (fewer data points)
- Use high contrast

### For Web/Demos

- Use `dpi=100` for faster loading
- Smaller figure sizes
- Interactive formats (Plotly) for exploration

---

## Testing

Run the built-in tests:

```python
python src/utils/visualization.py
```

This will generate all plot types with dummy data and verify functionality.

---

## Output Directory Structure

Recommended organization:

```
results/
└── figures/
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── comparison.png
    ├── attention.png
    ├── ablation.png
    └── summary_table.png
```

---

## Next Steps

1. Train models and collect results
2. Generate all visualizations
3. Include in PDF report
4. Use in demo notebook

---

For more examples, see `notebooks/CogniSense_Demo.ipynb`.

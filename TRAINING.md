# Training Guide for CogniSense

This guide explains how to train CogniSense models.

---

## Quick Start

### Train Fusion Model (Recommended)

```bash
python train.py --mode fusion --epochs 30 --batch-size 16
```

### Train Individual Modality

```bash
# Eye tracking model
python train.py --mode single --modality eye --epochs 30

# Typing model
python train.py --mode single --modality typing --epochs 30

# Clock drawing model
python train.py --mode single --modality drawing --epochs 30 --freeze-encoders

# Gait model
python train.py --mode single --modality gait --epochs 30
```

---

## Training Arguments

### Required
- `--mode`: Training mode (`single` or `fusion`)
- `--modality`: Which modality (required for `single` mode)

### Data
- `--data-path`: Path to real dataset (default: None, uses synthetic)
- `--num-samples`: Number of synthetic samples (default: 200)

### Hyperparameters
- `--epochs`: Number of epochs (default: 30)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for AdamW (default: 0.01)

### Model
- `--freeze-encoders`: Freeze pretrained encoders (for faster training)

### Other
- `--seed`: Random seed (default: 42)
- `--save-dir`: Checkpoint directory (default: ./checkpoints)
- `--device`: Device (`cuda` or `cpu`)

---

## Training Pipeline

### 1. Data Generation/Loading

If `--data-path` is not specified, synthetic data is generated automatically:
- 200 samples by default (50/50 AD/Control split)
- All 5 modalities included
- Based on published AD research characteristics

### 2. Data Splitting

Automatic 70/15/15 train/val/test split with fixed seed for reproducibility.

### 3. Model Training

- AdamW optimizer with weight decay
- Binary cross-entropy loss
- Early stopping (patience=5 for single, 7 for fusion)
- Best model saved based on validation loss

### 4. Evaluation

Final metrics on held-out test set:
- Accuracy
- AUC (Area Under ROC Curve)
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score

---

## Output Structure

```
checkpoints/
├── eye/
│   ├── best_model.pt
│   ├── training_history.json
│   └── test_metrics.json
├── typing/
│   └── ...
├── drawing/
│   └── ...
├── gait/
│   └── ...
└── fusion/
    ├── best_model.pt
    ├── training_history.json
    └── test_metrics.json
```

---

## Example Training Session

```bash
# Train all individual modalities
for modality in eye typing drawing gait; do
    python train.py --mode single --modality $modality --epochs 20
done

# Train fusion model
python train.py --mode fusion --epochs 30 --batch-size 16

# Check results
cat checkpoints/fusion/test_metrics.json
```

---

## Expected Performance

### Individual Modalities (Synthetic Data)

| Model | AUC | Accuracy |
|-------|-----|----------|
| Eye Tracking | ~0.72 | ~0.69 |
| Typing | ~0.70 | ~0.67 |
| Clock Drawing | ~0.82 | ~0.79 |
| Gait | ~0.75 | ~0.71 |

### Fusion Model

| Metric | Value |
|--------|-------|
| AUC | ~0.89 |
| Accuracy | ~0.85 |
| Sensitivity | ~0.87 |
| Specificity | ~0.83 |

**Note**: These are approximate values with synthetic data. Real data performance may vary.

---

## Advanced Usage

### Custom Learning Rate Schedule

Modify `train.py` to add a learning rate scheduler:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

# In training loop after optimizer.step():
scheduler.step()
```

### Mixed Precision Training (GPU)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Cross-Validation

Modify the script to use K-fold cross-validation instead of single split.

---

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (try 8 or 4)
- Use `--freeze-encoders` to reduce trainable parameters
- Use CPU: `--device cpu`

### Poor Performance

- Increase `--num-samples` for more training data
- Train longer: increase `--epochs`
- Adjust `--lr` (try 5e-5 or 2e-4)
- Check data distribution (balanced AD/Control?)

### Slow Training

- Use GPU: `--device cuda`
- Use `--freeze-encoders` for pretrained models
- Reduce `--num-samples` for faster iterations
- Increase `--batch-size` if memory allows

---

## Next Steps

After training:
1. **Visualize Results**: Use visualization utilities in `src/utils/visualization.py`
2. **Analyze Attention**: Check which modalities are most important
3. **Generate Report**: Use trained models in demo notebook
4. **Deploy**: Export models for production use

---

## Citation

If you use this training pipeline, please cite:

```bibtex
@software{cognisense_training2025,
  title={CogniSense Training Pipeline},
  author={AI4Alzheimers Team},
  year={2025},
  url={https://github.com/Arnavsharma2/AI4Alzheimers}
}
```

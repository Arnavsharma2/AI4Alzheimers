# CogniSense Notebooks

## Available Notebooks

### 1. [CogniSense_Demo.ipynb](CogniSense_Demo.ipynb) ⭐ **Main Demo**
The primary demonstration notebook for the hackathon submission.

**Features:**
- Complete walkthrough of CogniSense system
- Interactive demo with synthetic data
- Model architecture explanation
- Performance metrics and visualizations
- Explainability analysis with attention weights

**Run in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arnavsharma2/AI4Alzheimers/blob/main/notebooks/CogniSense_Demo.ipynb)

### 2. [Experimental_Results.ipynb](Experimental_Results.ipynb)
Comprehensive experimental results and analysis.

**Contents:**
- Cross-validation results
- Ablation studies
- Modality comparison
- Performance across different metrics

### 3. [Functional_Testing.ipynb](Functional_Testing.ipynb)
Functional testing suite for all components.

**Tests:**
- Data generator validation
- Model architecture verification
- Inference pipeline testing
- Integration tests

### 4. [Test_Phase1.ipynb](Test_Phase1.ipynb)
Phase 1 validation tests.

**Coverage:**
- Individual modality models
- Data processing pipeline
- Basic functionality

### 5. [Test_Phase2.ipynb](Test_Phase2.ipynb)
Phase 2 validation tests.

**Coverage:**
- Multimodal fusion
- Attention mechanism
- End-to-end pipeline

## Quick Start in Google Colab

### Option 1: Direct Upload
1. Go to [Google Colab](https://colab.research.google.com)
2. Upload the notebook file
3. Run cells sequentially

### Option 2: From GitHub
1. Open Colab
2. Click "File" → "Open notebook"
3. Select "GitHub" tab
4. Enter: `Arnavsharma2/AI4Alzheimers`
5. Select the desired notebook

## Notebook Structure

All notebooks follow this structure:

```
1. Setup & Installation
   - Install dependencies
   - Clone repository
   - Import libraries

2. Data Generation/Loading
   - Generate synthetic data
   - Visualize samples

3. Model Initialization
   - Load model architectures
   - Initialize fusion system

4. Experiments/Demo
   - Run inference
   - Generate results

5. Visualization & Analysis
   - Plot results
   - Explainability
```

## Requirements

All notebooks automatically install required dependencies:
- PyTorch 2.0+
- Transformers (Hugging Face)
- Gradio
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, SciPy
- SHAP, Plotly

## Colab Compatibility

✅ All notebooks are **fully compatible** with Google Colab:
- Auto-install dependencies
- No local files required
- Clones repository from GitHub
- Uses CPU-friendly model configurations
- Works on free Colab tier

## Expected Runtime

| Notebook | First Run | Subsequent Runs |
|----------|-----------|-----------------|
| CogniSense_Demo | ~5-8 min | ~2-3 min |
| Experimental_Results | ~10-15 min | ~5-7 min |
| Functional_Testing | ~3-5 min | ~1-2 min |
| Test_Phase1 | ~4-6 min | ~2-3 min |
| Test_Phase2 | ~6-9 min | ~3-4 min |

*First run includes model downloads (~1-2GB)*

## Troubleshooting

### "Module not found" error
Run the installation cell again:
```python
!pip install torch transformers gradio
```

### "Repository not found" error
Update the git clone command with correct repo URL.

### Out of memory error
Use Colab Pro or reduce batch sizes in training cells.

## Notes

- **GPU not required** - All notebooks work on CPU (optimized for Colab free tier)
- **Synthetic data** - Demo uses generated data for reproducibility
- **Pretrained models** - Downloads from Hugging Face on first run
- **Deterministic** - Random seeds set for reproducible results

## Support

For issues with notebooks:
1. Check the main [README.md](../README.md)
2. See [QUICKSTART.md](../QUICKSTART.md)
3. Open an issue on GitHub

---

**Recommended:** Start with [CogniSense_Demo.ipynb](CogniSense_Demo.ipynb) for the complete experience!

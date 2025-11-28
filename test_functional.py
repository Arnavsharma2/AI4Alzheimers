#!/usr/bin/env python3
"""
Functional integration tests for CogniSense.
Tests actual model instantiation and forward passes.
Requires PyTorch and other ML dependencies to be installed.
"""

import sys
import numpy as np

def test_numpy():
    """Test NumPy is available and working."""
    print("\n" + "="*60)
    print("  TEST 1: NumPy Functionality")
    print("="*60)

    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        print(f"✓ NumPy version: {np.__version__}")
        print("✓ NumPy operations working")
        return True
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_torch_import():
    """Test PyTorch can be imported."""
    print("\n" + "="*60)
    print("  TEST 2: PyTorch Import")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        assert z.shape == (2, 4)
        print("✓ Basic tensor operations working")
        return True
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_synthetic_data_generation():
    """Test synthetic data generators."""
    print("\n" + "="*60)
    print("  TEST 3: Synthetic Data Generation")
    print("="*60)

    try:
        from src.data_processing.synthetic_data_generator import (
            EyeTrackingGenerator,
            TypingDynamicsGenerator,
            ClockDrawingGenerator,
            GaitDataGenerator
        )

        # Test eye tracking
        eye_gen = EyeTrackingGenerator()
        eye_data = eye_gen.generate_sequence(is_alzheimers=True)
        assert eye_data.shape[1] == 2  # x, y coordinates
        print(f"✓ Eye tracking data: {eye_data.shape}")

        # Test typing
        typing_gen = TypingDynamicsGenerator()
        typing_data = typing_gen.generate_sequence(is_alzheimers=True)
        assert typing_data.shape[1] == 5  # Features
        print(f"✓ Typing data: {typing_data.shape}")

        # Test drawing
        drawing_gen = ClockDrawingGenerator()
        drawing_data = drawing_gen.generate_image(is_alzheimers=True)
        assert drawing_data.shape == (224, 224, 3)
        print(f"✓ Clock drawing image: {drawing_data.shape}")

        # Test gait
        gait_gen = GaitDataGenerator()
        gait_data = gait_gen.generate_sequence(is_alzheimers=True)
        assert gait_data.shape[1] == 3  # x, y, z acceleration
        print(f"✓ Gait data: {gait_data.shape}")

        print("✓ All synthetic data generators working")
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_instantiation():
    """Test that all models can be instantiated."""
    print("\n" + "="*60)
    print("  TEST 4: Model Instantiation")
    print("="*60)

    try:
        import torch
        from src.models.eye_model import EyeTrackingModel
        from src.models.typing_model import TypingModel
        from src.models.drawing_model import ClockDrawingModel
        from src.models.gait_model import GaitModel

        # Eye model
        eye_model = EyeTrackingModel()
        print(f"✓ EyeTrackingModel created")

        # Typing model
        typing_model = TypingModel()
        print(f"✓ TypingModel created")

        # Drawing model (may take time to download)
        print("  Loading ClockDrawingModel (may download pretrained weights)...")
        drawing_model = ClockDrawingModel()
        print(f"✓ ClockDrawingModel created")

        # Gait model
        gait_model = GaitModel()
        print(f"✓ GaitModel created")

        print("✓ All individual models instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward_pass():
    """Test forward passes through models."""
    print("\n" + "="*60)
    print("  TEST 5: Model Forward Pass")
    print("="*60)

    try:
        import torch
        from src.models.eye_model import EyeTrackingModel
        from src.models.typing_model import TypingModel
        from src.models.gait_model import GaitModel
        from src.data_processing.synthetic_data_generator import (
            EyeTrackingGenerator,
            TypingDynamicsGenerator,
            GaitDataGenerator
        )

        # Eye model
        eye_gen = EyeTrackingGenerator()
        eye_data = eye_gen.generate_sequence(is_alzheimers=True)
        eye_model = EyeTrackingModel()
        eye_model.eval()
        with torch.no_grad():
            eye_input = torch.FloatTensor(eye_data).unsqueeze(0)
            eye_output = eye_model(eye_input)
            assert eye_output.shape == (1, 64)
            print(f"✓ Eye model forward pass: {eye_input.shape} → {eye_output.shape}")

        # Typing model
        typing_gen = TypingDynamicsGenerator()
        typing_data = typing_gen.generate_sequence(is_alzheimers=True)
        typing_model = TypingModel()
        typing_model.eval()
        with torch.no_grad():
            typing_input = torch.FloatTensor(typing_data).unsqueeze(0)
            typing_output = typing_model(typing_input)
            assert typing_output.shape == (1, 64)
            print(f"✓ Typing model forward pass: {typing_input.shape} → {typing_output.shape}")

        # Gait model
        gait_gen = GaitDataGenerator()
        gait_data = gait_gen.generate_sequence(is_alzheimers=True)
        gait_model = GaitModel()
        gait_model.eval()
        with torch.no_grad():
            gait_input = torch.FloatTensor(gait_data).unsqueeze(0).permute(0, 2, 1)
            gait_output = gait_model(gait_input)
            assert gait_output.shape == (1, 64)
            print(f"✓ Gait model forward pass: {gait_input.shape} → {gait_output.shape}")

        print("✓ All model forward passes successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_model():
    """Test multimodal fusion model."""
    print("\n" + "="*60)
    print("  TEST 6: Multimodal Fusion")
    print("="*60)

    try:
        import torch
        from src.fusion.fusion_model import MultimodalFusionModel
        from src.data_processing.synthetic_data_generator import (
            EyeTrackingGenerator,
            TypingDynamicsGenerator,
            ClockDrawingGenerator,
            GaitDataGenerator
        )

        # Generate sample data
        eye_gen = EyeTrackingGenerator()
        typing_gen = TypingDynamicsGenerator()
        drawing_gen = ClockDrawingGenerator()
        gait_gen = GaitDataGenerator()

        eye_data = eye_gen.generate_sequence(is_alzheimers=True)
        typing_data = typing_gen.generate_sequence(is_alzheimers=True)
        drawing_data = drawing_gen.generate_image(is_alzheimers=True)
        gait_data = gait_gen.generate_sequence(is_alzheimers=True)

        # Create fusion model
        print("  Creating MultimodalFusionModel...")
        fusion_model = MultimodalFusionModel(fusion_type='attention')
        fusion_model.eval()

        # Prepare inputs
        eye_input = torch.FloatTensor(eye_data).unsqueeze(0)
        typing_input = torch.FloatTensor(typing_data).unsqueeze(0)
        drawing_input = torch.FloatTensor(drawing_data).permute(2, 0, 1).unsqueeze(0) / 255.0
        gait_input = torch.FloatTensor(gait_data).unsqueeze(0).permute(0, 2, 1)

        # Forward pass
        with torch.no_grad():
            output, attention = fusion_model(
                speech_audio=None,
                speech_text=None,
                eye_tracking=eye_input,
                typing_dynamics=typing_input,
                clock_drawing=drawing_input,
                gait_data=gait_input,
                return_attention=True
            )

            assert output.shape == (1, 1)
            assert attention.shape == (1, 5)
            assert torch.allclose(attention.sum(), torch.tensor(1.0), atol=1e-6)

            print(f"✓ Fusion model output: {output.shape}")
            print(f"✓ Attention weights: {attention.shape}")
            print(f"✓ Attention sums to 1.0: {attention.sum().item():.6f}")
            print(f"  Risk score: {torch.sigmoid(output).item():.4f}")

        print("✓ Multimodal fusion model working")
        return True
    except Exception as e:
        print(f"✗ Fusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_utilities():
    """Test training utilities."""
    print("\n" + "="*60)
    print("  TEST 7: Training Utilities")
    print("="*60)

    try:
        import numpy as np
        from src.utils.training_utils import compute_metrics, EarlyStopping

        # Test metrics computation
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.2])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auc'] <= 1

        print(f"✓ Metrics computed: {list(metrics.keys())}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

        # Test early stopping
        early_stopping = EarlyStopping(patience=3, mode='max')

        # Simulate improving scores
        assert not early_stopping(0.75)
        assert not early_stopping(0.80)
        assert not early_stopping(0.85)
        print("✓ Early stopping tracking improvements")

        # Simulate no improvement
        assert not early_stopping(0.84)  # Slight decrease
        assert not early_stopping(0.83)  # Continue decreasing
        assert not early_stopping(0.82)  # Still patient
        should_stop = early_stopping(0.81)  # Patience exhausted
        print(f"✓ Early stopping triggered after patience: {should_stop}")

        print("✓ Training utilities working")
        return True
    except Exception as e:
        print(f"✗ Training utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    """Test PyTorch dataset creation."""
    print("\n" + "="*60)
    print("  TEST 8: Dataset Creation")
    print("="*60)

    try:
        import torch
        from torch.utils.data import DataLoader
        from src.data_processing.dataset import MultimodalAlzheimerDataset
        from src.data_processing.synthetic_data_generator import generate_synthetic_dataset

        # Generate small synthetic dataset
        print("  Generating synthetic dataset...")
        data_dict = generate_synthetic_dataset(num_samples=10, modalities=['eye', 'typing', 'gait'])

        # Create dataset
        dataset = MultimodalAlzheimerDataset(data_dict, modalities=['eye', 'typing', 'gait'])
        assert len(dataset) == 10
        print(f"✓ Dataset created with {len(dataset)} samples")

        # Test __getitem__
        sample = dataset[0]
        assert 'eye' in sample
        assert 'typing' in sample
        assert 'gait' in sample
        assert 'label' in sample
        print(f"✓ Sample keys: {list(sample.keys())}")

        # Test DataLoader
        from src.data_processing.dataset import custom_collate_fn
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

        batch = next(iter(dataloader))
        assert len(batch['eye']) == 4
        assert batch['label'].shape == (4,)
        print(f"✓ DataLoader batching working, batch size: {len(batch['eye'])}")

        print("✓ Dataset and DataLoader working")
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all functional tests."""
    print("\n" + "="*60)
    print("  🧠 CogniSense Functional Integration Tests")
    print("="*60)

    tests = [
        ("NumPy", test_numpy),
        ("PyTorch Import", test_torch_import),
        ("Data Generation", test_synthetic_data_generation),
        ("Model Instantiation", test_model_instantiation),
        ("Forward Passes", test_model_forward_pass),
        ("Fusion Model", test_fusion_model),
        ("Training Utils", test_training_utilities),
        ("Dataset Creation", test_dataset_creation),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

        # Stop if critical test fails
        if not results[name] and name in ["NumPy", "PyTorch Import"]:
            print(f"\n⚠️  Critical test '{name}' failed. Stopping further tests.")
            break

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:25s}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("\n" + "="*60)
    if total_passed == total_tests:
        print(f"  ✅ ALL {total_tests} TESTS PASSED")
        print("="*60)
        print("\n🎉 CogniSense is fully functional and ready!")
        return 0
    else:
        print(f"  ⚠️  {total_passed}/{total_tests} TESTS PASSED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

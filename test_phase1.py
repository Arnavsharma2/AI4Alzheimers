"""
Comprehensive Test Suite for CogniSense
Tests all components to ensure Phase 1 is functional

Run this in Google Colab after installing dependencies
"""

import sys
import traceback

def test_section(name):
    """Decorator for test sections"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

def run_tests():
    """Run all tests"""

    test_section("TEST 1: Import Dependencies")
    try:
        import torch
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✓ Core dependencies imported")
    except Exception as e:
        print(f"✗ Failed to import dependencies: {e}")
        return False

    test_section("TEST 2: Import CogniSense Models")
    try:
        from src.models.speech_model import SpeechModel
        from src.models.eye_model import EyeTrackingModel
        from src.models.typing_model import TypingModel
        from src.models.drawing_model import ClockDrawingModel
        from src.models.gait_model import GaitModel
        from src.fusion.fusion_model import MultimodalFusionModel
        print("✓ All model classes imported successfully")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 3: Import Data Generators")
    try:
        from src.data_processing.synthetic_data_generator import (
            EyeTrackingGenerator,
            TypingDynamicsGenerator,
            ClockDrawingGenerator,
            GaitDataGenerator,
            generate_synthetic_dataset
        )
        print("✓ All data generators imported successfully")
    except Exception as e:
        print(f"✗ Failed to import generators: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 4: Generate Synthetic Data")
    try:
        # Eye tracking
        eye_gen = EyeTrackingGenerator()
        eye_data = eye_gen.generate_sequence(is_alzheimers=False)
        assert eye_data.shape == (100, 2), f"Eye data shape mismatch: {eye_data.shape}"
        print(f"✓ Eye tracking data: {eye_data.shape}")

        # Typing
        typing_gen = TypingDynamicsGenerator()
        typing_data = typing_gen.generate_sequence(is_alzheimers=True)
        assert typing_data.shape == (50, 5), f"Typing data shape mismatch: {typing_data.shape}"
        print(f"✓ Typing data: {typing_data.shape}")

        # Clock drawing
        clock_gen = ClockDrawingGenerator(image_size=224)
        clock_img = clock_gen.generate_image(is_alzheimers=False)
        assert clock_img.size == (224, 224), f"Clock image size mismatch: {clock_img.size}"
        print(f"✓ Clock drawing: {clock_img.size}")

        # Gait
        gait_gen = GaitDataGenerator()
        gait_data = gait_gen.generate_sequence(is_alzheimers=False)
        assert gait_data.shape[0] == 3, f"Gait data channels mismatch: {gait_data.shape}"
        print(f"✓ Gait data: {gait_data.shape}")

    except Exception as e:
        print(f"✗ Failed to generate synthetic data: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 5: Instantiate Individual Models")
    try:
        # Speech model (with freeze to avoid downloading in test)
        speech_model = SpeechModel(freeze_encoders=True, output_dim=64)
        print(f"✓ Speech model: {sum(p.numel() for p in speech_model.parameters()):,} params")

        # Eye model
        eye_model = EyeTrackingModel(output_dim=64)
        print(f"✓ Eye model: {sum(p.numel() for p in eye_model.parameters()):,} params")

        # Typing model
        typing_model = TypingModel(output_dim=64)
        print(f"✓ Typing model: {sum(p.numel() for p in typing_model.parameters()):,} params")

        # Clock drawing model
        drawing_model = ClockDrawingModel(freeze_encoder=True, output_dim=64)
        print(f"✓ Clock drawing model: {sum(p.numel() for p in drawing_model.parameters()):,} params")

        # Gait model
        gait_model = GaitModel(output_dim=64)
        print(f"✓ Gait model: {sum(p.numel() for p in gait_model.parameters()):,} params")

    except Exception as e:
        print(f"✗ Failed to instantiate models: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 6: Instantiate Fusion Model")
    try:
        fusion_model = MultimodalFusionModel(
            speech_config={'freeze_encoders': True},
            drawing_config={'freeze_encoder': True},
            fusion_type='attention'
        )
        total_params = sum(p.numel() for p in fusion_model.parameters())
        print(f"✓ Fusion model instantiated: {total_params:,} total params")
        fusion_model.eval()
        print("✓ Model set to eval mode")
    except Exception as e:
        print(f"✗ Failed to instantiate fusion model: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 7: Test Forward Pass (Small Dummy Data)")
    try:
        batch_size = 2

        # Create dummy inputs
        from transformers import BertTokenizer, ViTImageProcessor

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        # Speech
        dummy_audio = torch.randn(batch_size, 16000)
        dummy_text = ["test sentence"] * batch_size
        speech_audio = {'input_values': dummy_audio}
        speech_text = bert_tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=True)

        # Eye
        eye_tensor = torch.FloatTensor(batch_size, 100, 2).uniform_(0, 1)

        # Typing
        typing_tensor = torch.FloatTensor(batch_size, 50, 5).uniform_(0, 0.5)

        # Drawing (use generated clock image)
        from PIL import Image
        clock_imgs = [clock_gen.generate_image(False) for _ in range(batch_size)]
        clock_processed = vit_processor(images=clock_imgs, return_tensors="pt")
        drawing_tensor = clock_processed['pixel_values']

        # Gait
        gait_tensor = torch.FloatTensor(batch_size, 3, 500).uniform_(-10, 10)

        print("✓ Dummy inputs created")

        # Forward pass
        with torch.no_grad():
            risk_score, attention_weights, modality_features = fusion_model(
                speech_audio=speech_audio,
                speech_text=speech_text,
                eye_gaze=eye_tensor,
                typing_sequence=typing_tensor,
                drawing_image=drawing_tensor,
                gait_sensor=gait_tensor,
                return_attention=True,
                return_modality_features=True
            )

        print(f"✓ Forward pass successful!")
        print(f"  Risk scores: {risk_score.squeeze().tolist()}")
        print(f"  Attention weights shape: {attention_weights.shape}")
        print(f"  Modality features: {list(modality_features.keys())}")

        # Validate outputs
        assert risk_score.shape == (batch_size, 1), f"Risk score shape mismatch: {risk_score.shape}"
        assert attention_weights.shape == (batch_size, 5), f"Attention shape mismatch: {attention_weights.shape}"
        assert len(modality_features) == 5, f"Expected 5 modalities, got {len(modality_features)}"

        # Check attention weights sum to ~1
        attention_sum = attention_weights.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones(batch_size), atol=1e-5), \
            f"Attention weights don't sum to 1: {attention_sum}"

        print("✓ Output validation passed")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 8: Test Full Dataset Generation")
    try:
        dataset = generate_synthetic_dataset(num_samples=20, ad_ratio=0.5)
        print(f"✓ Generated {len(dataset['labels'])} samples")
        print(f"  AD samples: {sum(dataset['labels'])}")
        print(f"  Control samples: {len(dataset['labels']) - sum(dataset['labels'])}")

        # Validate dataset structure
        assert 'eye_tracking' in dataset
        assert 'typing' in dataset
        assert 'clock_drawing' in dataset
        assert 'gait' in dataset
        assert 'labels' in dataset
        print("✓ Dataset structure valid")

    except Exception as e:
        print(f"✗ Dataset generation failed: {e}")
        traceback.print_exc()
        return False

    test_section("TEST 9: Test Visualization Functions")
    try:
        import matplotlib.pyplot as plt

        # Test simple plot
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        plt.close(fig)
        print("✓ Matplotlib working")

        # Test with actual data
        eye_sample = dataset['eye_tracking'][0]
        fig, ax = plt.subplots()
        ax.plot(eye_sample[:, 0], eye_sample[:, 1])
        plt.close(fig)
        print("✓ Can plot eye tracking data")

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        traceback.print_exc()
        return False

    test_section("✅ ALL TESTS PASSED!")
    print("\nPhase 1 is fully functional and ready for submission!")
    print("\nNext steps:")
    print("  1. Open notebooks/CogniSense_Demo.ipynb in Google Colab")
    print("  2. Run all cells to generate interactive demo")
    print("  3. Test with both AD and Control samples")
    print("  4. Proceed to Phase 2 (Training Scripts)")

    return True


if __name__ == "__main__":
    print("🧠 CogniSense Phase 1 Test Suite")
    print("=" * 60)

    success = run_tests()

    if success:
        print("\n" + "=" * 60)
        print("  ✅ PHASE 1: COMPLETE AND FUNCTIONAL")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("  ❌ PHASE 1: TESTS FAILED")
        print("=" * 60)
        sys.exit(1)

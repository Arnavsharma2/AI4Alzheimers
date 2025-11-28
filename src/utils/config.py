"""
Configuration settings for CogniSense
"""

import os
from pathlib import Path

class Config:
    """Global configuration for CogniSense project"""

    # Project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    RESULTS_DIR = ROOT_DIR / "results"

    # Data paths by modality
    SPEECH_DATA = RAW_DATA_DIR / "speech"
    EYE_DATA = RAW_DATA_DIR / "eye_tracking"
    TYPING_DATA = RAW_DATA_DIR / "typing"
    DRAWING_DATA = RAW_DATA_DIR / "clock_drawing"
    GAIT_DATA = RAW_DATA_DIR / "gait"

    # Model configuration
    SEED = 42
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Speech model config
    SPEECH_CONFIG = {
        'wav2vec_model': 'facebook/wav2vec2-base',
        'bert_model': 'bert-base-uncased',
        'sample_rate': 16000,
        'max_duration': 60,  # seconds
        'feature_dim': 768
    }

    # Eye tracking config
    EYE_CONFIG = {
        'sequence_length': 100,
        'sampling_rate': 60,  # Hz
        'feature_dim': 2,  # x, y coordinates
        'hidden_dim': 128
    }

    # Typing config
    TYPING_CONFIG = {
        'sequence_length': 50,
        'feature_dim': 5,  # flight, dwell, digraph, error_rate, pause
        'hidden_dim': 128
    }

    # Clock drawing config
    DRAWING_CONFIG = {
        'image_size': 224,
        'vit_model': 'google/vit-base-patch16-224',
        'feature_dim': 768
    }

    # Gait config
    GAIT_CONFIG = {
        'window_size': 128,
        'sampling_rate': 50,  # Hz
        'channels': 3,  # x, y, z accelerometer
        'feature_dim': 256
    }

    # Fusion model config
    FUSION_CONFIG = {
        'input_dim': 64,  # Feature dimension from each modality
        'num_modalities': 5,
        'hidden_dim': 256,
        'dropout': 0.4
    }

    # Training config
    TRAINING_CONFIG = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 30,
        'num_folds': 5,
        'early_stopping_patience': 5
    }

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    @classmethod
    def create_dirs(cls):
        """Create all necessary directories"""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR, cls.RESULTS_DIR,
            cls.SPEECH_DATA, cls.EYE_DATA, cls.TYPING_DATA,
            cls.DRAWING_DATA, cls.GAIT_DATA
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

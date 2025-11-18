"""
Neural network models for each modality
"""

from .speech_model import SpeechModel
from .eye_model import EyeTrackingModel
from .typing_model import TypingModel
from .drawing_model import ClockDrawingModel
from .gait_model import GaitModel

__all__ = [
    'SpeechModel',
    'EyeTrackingModel',
    'TypingModel',
    'ClockDrawingModel',
    'GaitModel'
]

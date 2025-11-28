"""
Data processing modules for each modality
"""

from .speech_processor import SpeechProcessor
from .eye_processor import EyeTrackingProcessor
from .typing_processor import TypingProcessor
from .drawing_processor import DrawingProcessor
from .gait_processor import GaitProcessor

__all__ = [
    'SpeechProcessor',
    'EyeTrackingProcessor',
    'TypingProcessor',
    'DrawingProcessor',
    'GaitProcessor'
]

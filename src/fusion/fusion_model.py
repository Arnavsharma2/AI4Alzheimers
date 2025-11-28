"""
Multimodal Fusion Model - Combines all 5 modalities with attention mechanism
"""

import torch
import torch.nn as nn
from ..models.speech_model import SpeechModel
from ..models.eye_model import EyeTrackingModel
from ..models.typing_model import TypingModel
from ..models.drawing_model import ClockDrawingModel
from ..models.gait_model import GaitModel

class MultimodalFusionModel(nn.Module):
    """
    CogniSense Multimodal Fusion Architecture

    Combines 5 modalities with learned attention weights:
    1. Speech (acoustic + linguistic)
    2. Eye tracking (gaze patterns)
    3. Typing dynamics (keystroke patterns)
    4. Clock drawing (visuospatial)
    5. Gait (movement patterns)

    Uses late fusion with cross-modal attention.
    """

    def __init__(
        self,
        speech_config=None,
        eye_config=None,
        typing_config=None,
        drawing_config=None,
        gait_config=None,
        freeze_modality_encoders=False,
        fusion_type="attention"  # "attention", "concat", "weighted"
    ):
        super(MultimodalFusionModel, self).__init__()

        self.fusion_type = fusion_type

        # Initialize individual modality models
        self.speech_model = SpeechModel(**(speech_config or {}))
        self.eye_model = EyeTrackingModel(**(eye_config or {}))
        self.typing_model = TypingModel(**(typing_config or {}))
        self.drawing_model = ClockDrawingModel(**(drawing_config or {}))
        self.gait_model = GaitModel(**(gait_config or {}))

        # Freeze modality encoders if specified (after pre-training)
        if freeze_modality_encoders:
            self._freeze_encoders()

        # Feature dimension from each modality
        feature_dim = 64
        num_modalities = 5

        # Cross-modal attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, num_modalities),
            nn.Softmax(dim=1)
        )

        # Fusion layers
        if fusion_type == "attention":
            # Attention-weighted fusion
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * num_modalities, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            )
        elif fusion_type == "concat":
            # Simple concatenation fusion
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * num_modalities, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _freeze_encoders(self):
        """Freeze all modality encoder weights"""
        for param in self.speech_model.parameters():
            param.requires_grad = False
        for param in self.eye_model.parameters():
            param.requires_grad = False
        for param in self.typing_model.parameters():
            param.requires_grad = False
        for param in self.drawing_model.parameters():
            param.requires_grad = False
        for param in self.gait_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        speech_audio=None,
        speech_text=None,
        eye_gaze=None,
        typing_sequence=None,
        drawing_image=None,
        gait_sensor=None,
        return_attention=False,
        return_modality_features=False
    ):
        """
        Forward pass through all modalities

        Args:
            speech_audio: Audio input for speech model
            speech_text: Text input for speech model
            eye_gaze: Gaze sequence for eye tracking model
            typing_sequence: Keystroke sequence for typing model
            drawing_image: Clock drawing image
            gait_sensor: Accelerometer data for gait model
            return_attention: If True, return attention weights
            return_modality_features: If True, return individual modality features

        Returns:
            risk_score: (batch_size, 1) Alzheimer's risk score
            attention_weights: (batch_size, 5) if return_attention=True
            modality_features: dict of features if return_modality_features=True
        """
        batch_size = None
        features_list = []
        modality_features = {}

        # Extract features from each available modality
        if speech_audio is not None and speech_text is not None:
            speech_feat = self.speech_model(speech_audio, speech_text)
            features_list.append(speech_feat)
            modality_features['speech'] = speech_feat
            batch_size = speech_feat.size(0)

        if eye_gaze is not None:
            eye_feat = self.eye_model(eye_gaze)
            features_list.append(eye_feat)
            modality_features['eye'] = eye_feat
            batch_size = eye_feat.size(0)

        if typing_sequence is not None:
            typing_feat = self.typing_model(typing_sequence)
            features_list.append(typing_feat)
            modality_features['typing'] = typing_feat
            batch_size = typing_feat.size(0)

        if drawing_image is not None:
            drawing_feat = self.drawing_model(drawing_image)
            features_list.append(drawing_feat)
            modality_features['drawing'] = drawing_feat
            batch_size = drawing_feat.size(0)

        if gait_sensor is not None:
            gait_feat = self.gait_model(gait_sensor)
            features_list.append(gait_feat)
            modality_features['gait'] = gait_feat
            batch_size = gait_feat.size(0)

        if len(features_list) == 0:
            raise ValueError("At least one modality input must be provided")

        # Concatenate all features
        all_features = torch.cat(features_list, dim=1)  # (batch, feature_dim * num_modalities)

        # Calculate attention weights
        attention_weights = self.attention(all_features)  # (batch, num_modalities)

        # Apply attention (weighted sum of individual features)
        if self.fusion_type == "attention":
            # Reshape features for attention
            features_reshaped = torch.stack(features_list, dim=1)  # (batch, num_modalities, feature_dim)
            attention_expanded = attention_weights.unsqueeze(2)  # (batch, num_modalities, 1)

            # Weighted features
            weighted_features = features_reshaped * attention_expanded  # (batch, num_modalities, feature_dim)
            weighted_features_flat = weighted_features.view(batch_size, -1)  # (batch, feature_dim * num_modalities)

            # Fusion
            fused = self.fusion(weighted_features_flat)
        else:
            # Simple concatenation
            fused = self.fusion(all_features)

        # Final prediction
        risk_score = self.classifier(fused)  # (batch, 1)

        # Return based on flags
        outputs = [risk_score]
        if return_attention:
            outputs.append(attention_weights)
        if return_modality_features:
            outputs.append(modality_features)

        return tuple(outputs) if len(outputs) > 1 else risk_score


class CogniSense(MultimodalFusionModel):
    """
    Alias for MultimodalFusionModel with better branding
    """
    pass


if __name__ == "__main__":
    # Test fusion model
    print("Testing MultimodalFusionModel...")

    model = MultimodalFusionModel()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy inputs
    batch_size = 4

    speech_audio = {'input_values': torch.randn(batch_size, 16000)}
    speech_text = {
        'input_ids': torch.randint(0, 1000, (batch_size, 50)),
        'attention_mask': torch.ones(batch_size, 50)
    }
    eye_gaze = torch.randn(batch_size, 100, 2)
    typing_sequence = torch.randn(batch_size, 50, 5)
    drawing_image = torch.randn(batch_size, 3, 224, 224)
    gait_sensor = torch.randn(batch_size, 3, 128)

    # Forward pass
    risk_score, attention_weights, modality_features = model(
        speech_audio=speech_audio,
        speech_text=speech_text,
        eye_gaze=eye_gaze,
        typing_sequence=typing_sequence,
        drawing_image=drawing_image,
        gait_sensor=gait_sensor,
        return_attention=True,
        return_modality_features=True
    )

    print(f"Risk score shape: {risk_score.shape}")  # (4, 1)
    print(f"Attention weights shape: {attention_weights.shape}")  # (4, 5)
    print(f"Modality features: {modality_features.keys()}")
    print(f"Sample attention weights: {attention_weights[0]}")
    print("MultimodalFusionModel test passed!")

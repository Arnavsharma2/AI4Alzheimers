"""
Speech Analysis Model - Dual-path architecture
Combines acoustic features (Wav2Vec2) with linguistic features (BERT)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel

class SpeechModel(nn.Module):
    """
    Dual-path speech model for Alzheimer's detection
    - Acoustic branch: Wav2Vec 2.0 for audio features
    - Linguistic branch: BERT for text features
    - Fusion: Concatenate + MLP
    """

    def __init__(
        self,
        wav2vec_model="facebook/wav2vec2-base",
        bert_model="bert-base-uncased",
        freeze_encoders=False,
        output_dim=64
    ):
        super(SpeechModel, self).__init__()

        # Acoustic branch
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model)
        wav2vec_dim = self.wav2vec.config.hidden_size  # 768

        # Linguistic branch
        self.bert = BertModel.from_pretrained(bert_model)
        bert_dim = self.bert.config.hidden_size  # 768

        # Freeze encoders if specified (for faster training)
        if freeze_encoders:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False

        # Fusion layers
        combined_dim = wav2vec_dim + bert_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, audio_input, text_input):
        """
        Forward pass

        Args:
            audio_input: Dict with keys 'input_values' for wav2vec
            text_input: Dict with keys 'input_ids', 'attention_mask' for BERT

        Returns:
            features: (batch_size, output_dim) feature embeddings
        """
        # Acoustic features
        acoustic_outputs = self.wav2vec(**audio_input)
        acoustic_features = acoustic_outputs.last_hidden_state.mean(dim=1)  # (batch, 768)

        # Linguistic features
        linguistic_outputs = self.bert(**text_input)
        linguistic_features = linguistic_outputs.pooler_output  # (batch, 768)

        # Concatenate
        combined = torch.cat([acoustic_features, linguistic_features], dim=1)  # (batch, 1536)

        # Fusion
        features = self.fusion(combined)  # (batch, output_dim)

        return features

class SpeechClassifier(nn.Module):
    """
    Complete speech-based classifier (for standalone training)
    """

    def __init__(self, speech_model_config=None):
        super(SpeechClassifier, self).__init__()

        if speech_model_config is None:
            speech_model_config = {}

        self.encoder = SpeechModel(**speech_model_config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_input, text_input):
        features = self.encoder(audio_input, text_input)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    # Test model
    print("Testing SpeechModel...")

    model = SpeechModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy inputs
    batch_size = 4
    audio_input = {
        'input_values': torch.randn(batch_size, 16000)  # 1 second at 16kHz
    }
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 50)),
        'attention_mask': torch.ones(batch_size, 50)
    }

    output = model(audio_input, text_input)
    print(f"Output shape: {output.shape}")  # (4, 64)
    print("SpeechModel test passed!")

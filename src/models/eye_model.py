"""
Eye Tracking Model - CNN-LSTM for gaze sequence analysis
"""

import torch
import torch.nn as nn

class EyeTrackingModel(nn.Module):
    """
    Eye tracking model for analyzing gaze patterns
    - CNN for local feature extraction
    - LSTM for temporal dependencies
    - Captures saccade patterns and fixation characteristics
    """

    def __init__(
        self,
        input_dim=2,  # x, y coordinates
        hidden_dim=128,
        num_layers=2,
        output_dim=64
    ):
        super(EyeTrackingModel, self).__init__()

        # 1D CNN for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, gaze_sequence):
        """
        Forward pass

        Args:
            gaze_sequence: (batch_size, sequence_length, 2) - (x, y) coordinates

        Returns:
            features: (batch_size, output_dim)
        """
        # Permute for CNN: (batch, 2, sequence_length)
        x = gaze_sequence.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn(x)  # (batch, 128, sequence_length/4)

        # Permute back for LSTM: (batch, sequence_length/4, 128)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        features = lstm_out[:, -1, :]  # (batch, hidden_dim*2)

        # Project to output dimension
        features = self.fc(features)  # (batch, output_dim)

        return features


class EyeTrackingClassifier(nn.Module):
    """
    Complete eye tracking-based classifier
    """

    def __init__(self, model_config=None):
        super(EyeTrackingClassifier, self).__init__()

        if model_config is None:
            model_config = {}

        self.encoder = EyeTrackingModel(**model_config)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, gaze_sequence):
        features = self.encoder(gaze_sequence)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    # Test model
    print("Testing EyeTrackingModel...")

    model = EyeTrackingModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input: batch of gaze sequences
    batch_size = 4
    sequence_length = 100
    gaze_input = torch.randn(batch_size, sequence_length, 2)

    output = model(gaze_input)
    print(f"Output shape: {output.shape}")  # (4, 64)
    print("EyeTrackingModel test passed!")

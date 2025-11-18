"""
Gait Analysis Model - 1D CNN for accelerometer data
"""

import torch
import torch.nn as nn

class GaitModel(nn.Module):
    """
    Gait analysis model for walking pattern analysis
    - 1D CNN for accelerometer/gyroscope data
    - Captures gait variability and balance issues
    """

    def __init__(
        self,
        input_channels=3,  # x, y, z accelerometer
        output_dim=64
    ):
        super(GaitModel, self).__init__()

        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, sensor_data):
        """
        Forward pass

        Args:
            sensor_data: (batch_size, channels, time_steps) - accelerometer data

        Returns:
            features: (batch_size, output_dim)
        """
        # CNN feature extraction
        x = self.conv_layers(sensor_data)  # (batch, 512, 1)

        # Fully connected layers
        features = self.fc(x)  # (batch, output_dim)

        return features


class GaitClassifier(nn.Module):
    """
    Complete gait-based classifier
    """

    def __init__(self, model_config=None):
        super(GaitClassifier, self).__init__()

        if model_config is None:
            model_config = {}

        self.encoder = GaitModel(**model_config)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, sensor_data):
        features = self.encoder(sensor_data)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    # Test model
    print("Testing GaitModel...")

    model = GaitModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input: batch of accelerometer sequences
    batch_size = 4
    time_steps = 128
    gait_input = torch.randn(batch_size, 3, time_steps)

    output = model(gait_input)
    print(f"Output shape: {output.shape}")  # (4, 64)
    print("GaitModel test passed!")

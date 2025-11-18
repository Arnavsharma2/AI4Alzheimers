"""
Typing Dynamics Model - Bidirectional LSTM for keystroke analysis
"""

import torch
import torch.nn as nn

class TypingModel(nn.Module):
    """
    Typing dynamics model for analyzing keystroke patterns
    - Features: flight time, dwell time, digraph latency, error rate, pause duration
    - Bidirectional LSTM captures temporal dependencies
    """

    def __init__(
        self,
        input_dim=5,  # typing features
        hidden_dim=128,
        num_layers=2,
        output_dim=64
    ):
        super(TypingModel, self).__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Attention mechanism for important keystrokes
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, typing_sequence):
        """
        Forward pass

        Args:
            typing_sequence: (batch_size, sequence_length, input_dim)

        Returns:
            features: (batch_size, output_dim)
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(typing_sequence)  # (batch, seq_len, hidden*2)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden*2)

        # Output projection
        features = self.fc(context)  # (batch, output_dim)

        return features


class TypingClassifier(nn.Module):
    """
    Complete typing-based classifier
    """

    def __init__(self, model_config=None):
        super(TypingClassifier, self).__init__()

        if model_config is None:
            model_config = {}

        self.encoder = TypingModel(**model_config)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, typing_sequence):
        features = self.encoder(typing_sequence)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    # Test model
    print("Testing TypingModel...")

    model = TypingModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input: batch of typing sequences
    batch_size = 4
    sequence_length = 50
    typing_input = torch.randn(batch_size, sequence_length, 5)

    output = model(typing_input)
    print(f"Output shape: {output.shape}")  # (4, 64)
    print("TypingModel test passed!")

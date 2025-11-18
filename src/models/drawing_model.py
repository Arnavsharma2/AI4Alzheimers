"""
Clock Drawing Model - Vision Transformer for spatial/executive function assessment
"""

import torch
import torch.nn as nn
from transformers import ViTModel

class ClockDrawingModel(nn.Module):
    """
    Clock drawing test model using Vision Transformer
    - Analyzes spatial organization, number placement, hand positioning
    - Pre-trained ViT fine-tuned for AD detection
    """

    def __init__(
        self,
        vit_model="google/vit-base-patch16-224",
        freeze_encoder=False,
        output_dim=64
    ):
        super(ClockDrawingModel, self).__init__()

        # Vision Transformer
        self.vit = ViTModel.from_pretrained(vit_model)
        vit_dim = self.vit.config.hidden_size  # 768

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Additional layers for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, image):
        """
        Forward pass

        Args:
            image: (batch_size, 3, 224, 224) - RGB images

        Returns:
            features: (batch_size, output_dim)
        """
        # ViT feature extraction
        outputs = self.vit(pixel_values=image)
        pooled_output = outputs.pooler_output  # (batch, 768)

        # Additional processing
        features = self.classifier(pooled_output)  # (batch, output_dim)

        return features


class ClockDrawingClassifier(nn.Module):
    """
    Complete clock drawing-based classifier
    """

    def __init__(self, model_config=None):
        super(ClockDrawingClassifier, self).__init__()

        if model_config is None:
            model_config = {}

        self.encoder = ClockDrawingModel(**model_config)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        features = self.encoder(image)
        output = self.classifier(features)
        return output


if __name__ == "__main__":
    # Test model
    print("Testing ClockDrawingModel...")

    model = ClockDrawingModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input: batch of images
    batch_size = 4
    image_input = torch.randn(batch_size, 3, 224, 224)

    output = model(image_input)
    print(f"Output shape: {output.shape}")  # (4, 64)
    print("ClockDrawingModel test passed!")

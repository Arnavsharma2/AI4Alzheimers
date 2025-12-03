#!/usr/bin/env python3
"""
Launch CogniSense Interactive Demo

Run this script to start the Gradio web interface:
    python launch_demo.py
"""

import sys
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("🧠 CogniSense - Multimodal Alzheimer's Detection")
print("=" * 60)
print("\n⏳ Initializing system...")
print("   (First run may take 1-2 minutes to download models)")
print()

from demo import launch_demo

if __name__ == "__main__":
    print("✓ Models loaded successfully!")
    print("🚀 Starting web interface...\n")

    demo = launch_demo()
    demo.launch(
        share=True,  # Create public URL
        server_name="0.0.0.0",
        server_port=7860
    )

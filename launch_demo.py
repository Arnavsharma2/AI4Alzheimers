#!/usr/bin/env python3
"""
Launch CogniSense Interactive Demo

Run this script to start the Gradio web interface:
    python launch_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from demo import launch_demo

if __name__ == "__main__":
    print("🧠 Launching CogniSense Demo...")
    print("=" * 60)
    demo = launch_demo()
    demo.launch(
        share=True,  # Create public URL
        server_name="0.0.0.0",
        server_port=7860
    )

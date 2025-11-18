"""
Synthetic Data Generators for CogniSense Modalities

Generates realistic synthetic data for:
- Eye tracking
- Typing dynamics
- Clock drawings
- Gait patterns

Based on published research characteristics of Alzheimer's disease.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
import random


class EyeTrackingGenerator:
    """
    Generate synthetic eye tracking data with AD-characteristic patterns
    """

    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def generate_sequence(
        self,
        is_alzheimers=False,
        sequence_length=100,
        task_type="reading"
    ):
        """
        Generate gaze sequence

        AD characteristics:
        - Longer fixation durations
        - Slower saccade velocity
        - More erratic scan paths
        - Reduced ability to inhibit irrelevant stimuli
        """
        gaze_sequence = []

        # Starting position
        x, y = self.screen_width / 2, self.screen_height / 2

        for i in range(sequence_length):
            if task_type == "reading":
                # Simulate left-to-right reading
                if i % 20 == 0:  # Line break
                    x = 100
                    y += np.random.normal(60, 20 if is_alzheimers else 10)
                else:
                    # Horizontal movement (reading)
                    dx = np.random.normal(30, 15 if is_alzheimers else 5)
                    dy = np.random.normal(0, 10 if is_alzheimers else 3)

                    x += dx
                    y += dy

                # AD: More backward saccades (regressions)
                if is_alzheimers and random.random() < 0.3:
                    x -= np.random.uniform(50, 150)

            else:  # free_viewing
                # Random saccades
                saccade_amplitude = np.random.gamma(2, 50)
                if is_alzheimers:
                    saccade_amplitude *= np.random.uniform(0.7, 1.3)  # More variable

                angle = np.random.uniform(0, 2 * np.pi)
                dx = saccade_amplitude * np.cos(angle)
                dy = saccade_amplitude * np.sin(angle)

                x += dx
                y += dy

            # Keep within screen bounds
            x = np.clip(x, 0, self.screen_width)
            y = np.clip(y, 0, self.screen_height)

            # Normalize to [0, 1]
            gaze_sequence.append([x / self.screen_width, y / self.screen_height])

        return np.array(gaze_sequence, dtype=np.float32)


class TypingDynamicsGenerator:
    """
    Generate synthetic typing dynamics with AD characteristics
    """

    def __init__(self):
        # Normal typing statistics (milliseconds)
        self.normal_stats = {
            'flight_time_mean': 200,
            'flight_time_std': 50,
            'dwell_time_mean': 100,
            'dwell_time_std': 30,
            'error_rate': 0.05
        }

        # AD degradation factors
        self.ad_stats = {
            'flight_time_mean': 300,  # +50% slower
            'flight_time_std': 100,   # More variable
            'dwell_time_mean': 150,   # Longer key presses
            'dwell_time_std': 60,
            'error_rate': 0.15        # More errors
        }

    def generate_sequence(self, is_alzheimers=False, sequence_length=50):
        """
        Generate typing dynamics sequence

        Features:
        - Flight time: Time between key releases
        - Dwell time: How long key is pressed
        - Digraph latency: Time between successive key presses
        - Error rate: Frequency of backspace
        - Pause duration: Long pauses mid-word
        """
        stats = self.ad_stats if is_alzheimers else self.normal_stats

        sequence = []

        for i in range(sequence_length):
            # Flight time (inter-key interval)
            flight_time = max(0, np.random.normal(
                stats['flight_time_mean'],
                stats['flight_time_std']
            ))

            # Dwell time
            dwell_time = max(0, np.random.normal(
                stats['dwell_time_mean'],
                stats['dwell_time_std']
            ))

            # Digraph latency
            digraph = flight_time + dwell_time

            # Error indicator (1 if error, 0 otherwise)
            error = 1 if random.random() < stats['error_rate'] else 0

            # Pause duration (long pauses in AD)
            if is_alzheimers and random.random() < 0.2:
                pause = np.random.uniform(500, 2000)  # Long pause
            else:
                pause = 0

            # Normalize features (divide by 1000 for seconds)
            sequence.append([
                flight_time / 1000,
                dwell_time / 1000,
                digraph / 1000,
                error,
                pause / 1000
            ])

        return np.array(sequence, dtype=np.float32)


class ClockDrawingGenerator:
    """
    Generate synthetic clock drawing images with AD characteristics
    """

    def __init__(self, image_size=224):
        self.image_size = image_size

    def generate_image(self, is_alzheimers=False):
        """
        Generate clock drawing image

        Normal characteristics:
        - Circular contour
        - All 12 numbers present
        - Numbers evenly spaced
        - Hands pointing to correct time

        AD characteristics:
        - Irregular circle
        - Missing or repeated numbers
        - Number crowding on one side
        - Incorrect hand placement
        """
        # Create blank image
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)

        center_x = self.image_size // 2
        center_y = self.image_size // 2
        radius = self.image_size // 3

        # Draw circle
        if is_alzheimers and random.random() < 0.4:
            # Irregular circle
            ellipse_width = radius * np.random.uniform(0.8, 1.2)
            ellipse_height = radius * np.random.uniform(0.8, 1.2)
            draw.ellipse(
                [center_x - ellipse_width, center_y - ellipse_height,
                 center_x + ellipse_width, center_y + ellipse_height],
                outline='black', width=2
            )
        else:
            # Regular circle
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                outline='black', width=2
            )

        # Draw numbers
        numbers = list(range(1, 13))

        if is_alzheimers:
            # AD: Sometimes miss numbers or repeat
            if random.random() < 0.3:
                numbers.remove(random.choice(numbers))
            if random.random() < 0.2:
                numbers.append(random.choice(range(1, 13)))

        for i, num in enumerate(numbers):
            if is_alzheimers:
                # Number crowding on one side
                angle = np.pi / 2 - (2 * np.pi * num / 12)
                angle += np.random.uniform(-0.5, 0.5)  # More error

                # Sometimes cluster numbers
                if random.random() < 0.3:
                    angle = np.random.uniform(0, np.pi)

                num_radius = radius * np.random.uniform(0.6, 1.0)
            else:
                # Evenly spaced
                angle = np.pi / 2 - (2 * np.pi * num / 12)
                num_radius = radius * 0.8

            num_x = center_x + num_radius * np.cos(angle)
            num_y = center_y - num_radius * np.sin(angle)

            # Draw number (simplified - just position marker)
            draw.ellipse([num_x-3, num_y-3, num_x+3, num_y+3], fill='black')

        # Draw clock hands (pointing to 10:10 as in typical tests)
        if is_alzheimers and random.random() < 0.4:
            # Incorrect hand placement
            hour_angle = np.random.uniform(0, 2 * np.pi)
            minute_angle = np.random.uniform(0, 2 * np.pi)
        else:
            # Correct: 10:10
            hour_angle = np.pi / 2 - (2 * np.pi * (10 / 12))
            minute_angle = np.pi / 2 - (2 * np.pi * (10 / 60))

        # Hour hand
        hour_length = radius * 0.5
        hour_x = center_x + hour_length * np.cos(hour_angle)
        hour_y = center_y - hour_length * np.sin(hour_angle)
        draw.line([center_x, center_y, hour_x, hour_y], fill='black', width=3)

        # Minute hand
        minute_length = radius * 0.7
        minute_x = center_x + minute_length * np.cos(minute_angle)
        minute_y = center_y - minute_length * np.sin(minute_angle)
        draw.line([center_x, center_y, minute_x, minute_y], fill='black', width=2)

        return img


class GaitDataGenerator:
    """
    Generate synthetic gait (accelerometer) data with AD characteristics
    """

    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate

    def generate_sequence(self, is_alzheimers=False, duration=10):
        """
        Generate accelerometer sequence during walking

        AD characteristics:
        - Slower gait speed
        - Increased stride variability
        - Reduced stride length
        - More irregular rhythm
        """
        num_samples = duration * self.sampling_rate

        if is_alzheimers:
            # AD gait parameters
            step_frequency = np.random.uniform(0.8, 1.2)  # Slower, Hz
            stride_variability = 0.3  # High variability
        else:
            # Normal gait
            step_frequency = np.random.uniform(1.5, 2.0)  # Hz
            stride_variability = 0.1  # Low variability

        # Generate walking pattern
        t = np.linspace(0, duration, num_samples)

        # X-axis (forward-backward)
        x = np.sin(2 * np.pi * step_frequency * t)
        x += np.random.normal(0, 0.1 * (1 + stride_variability), num_samples)

        # Y-axis (lateral)
        y = np.sin(2 * np.pi * step_frequency * t + np.pi / 2) * 0.5
        y += np.random.normal(0, 0.15 * (1 + stride_variability), num_samples)

        # Z-axis (vertical)
        z = np.abs(np.sin(4 * np.pi * step_frequency * t)) + 1.0
        z += np.random.normal(0, 0.1 * (1 + stride_variability), num_samples)

        # Add gravity
        z += 9.8

        # Stack and transpose
        gait_data = np.stack([x, y, z], axis=0).astype(np.float32)

        return gait_data


def generate_synthetic_dataset(
    num_samples=100,
    ad_ratio=0.5,
    output_dir=None
):
    """
    Generate complete synthetic dataset for all modalities

    Args:
        num_samples: Total number of samples to generate
        ad_ratio: Proportion of AD samples (default 0.5 for balanced dataset)
        output_dir: Directory to save generated data

    Returns:
        dataset: Dict containing all modality data and labels
    """
    num_ad = int(num_samples * ad_ratio)
    num_control = num_samples - num_ad

    labels = [1] * num_ad + [0] * num_control
    random.shuffle(labels)

    # Initialize generators
    eye_gen = EyeTrackingGenerator()
    typing_gen = TypingDynamicsGenerator()
    clock_gen = ClockDrawingGenerator()
    gait_gen = GaitDataGenerator()

    dataset = {
        'eye_tracking': [],
        'typing': [],
        'clock_drawing': [],
        'gait': [],
        'labels': labels
    }

    print(f"Generating {num_samples} synthetic samples...")
    print(f"  - {num_ad} AD samples")
    print(f"  - {num_control} Control samples")

    for i, label in enumerate(labels):
        is_ad = bool(label)

        # Generate data for each modality
        eye_data = eye_gen.generate_sequence(is_alzheimers=is_ad)
        typing_data = typing_gen.generate_sequence(is_alzheimers=is_ad)
        clock_img = clock_gen.generate_image(is_alzheimers=is_ad)
        gait_data = gait_gen.generate_sequence(is_alzheimers=is_ad)

        dataset['eye_tracking'].append(eye_data)
        dataset['typing'].append(typing_data)
        dataset['clock_drawing'].append(clock_img)
        dataset['gait'].append(gait_data)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    print("Synthetic dataset generation complete!")

    return dataset


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing synthetic data generators...")

    # Test eye tracking
    eye_gen = EyeTrackingGenerator()
    eye_normal = eye_gen.generate_sequence(is_alzheimers=False)
    eye_ad = eye_gen.generate_sequence(is_alzheimers=True)
    print(f"Eye tracking - Normal: {eye_normal.shape}, AD: {eye_ad.shape}")

    # Test typing
    typing_gen = TypingDynamicsGenerator()
    typing_normal = typing_gen.generate_sequence(is_alzheimers=False)
    typing_ad = typing_gen.generate_sequence(is_alzheimers=True)
    print(f"Typing - Normal: {typing_normal.shape}, AD: {typing_ad.shape}")

    # Test clock drawing
    clock_gen = ClockDrawingGenerator()
    clock_normal = clock_gen.generate_image(is_alzheimers=False)
    clock_ad = clock_gen.generate_image(is_alzheimers=True)
    print(f"Clock - Normal: {clock_normal.size}, AD: {clock_ad.size}")

    # Test gait
    gait_gen = GaitDataGenerator()
    gait_normal = gait_gen.generate_sequence(is_alzheimers=False)
    gait_ad = gait_gen.generate_sequence(is_alzheimers=True)
    print(f"Gait - Normal: {gait_normal.shape}, AD: {gait_ad.shape}")

    print("\nAll generators working correctly!")

    # Generate small dataset
    dataset = generate_synthetic_dataset(num_samples=20)
    print(f"\nDataset sizes:")
    for key, value in dataset.items():
        if key != 'labels':
            print(f"  {key}: {len(value)} samples")

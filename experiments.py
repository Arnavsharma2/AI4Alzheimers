#!/usr/bin/env python3
"""
Hyperparameter experimentation framework for CogniSense.
Demonstrates systematic exploration expected in month-long research.

This script runs multiple experiments with different configurations
to find optimal hyperparameters and architectural choices.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import itertools
from datetime import datetime
from pathlib import Path

from train import train_single_modality
from src.data_processing.synthetic_data_generator import generate_synthetic_dataset


class ExperimentRunner:
    """
    Manages and executes multiple training experiments.
    """

    def __init__(self, output_dir='experiments/results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_experiment(self, config, experiment_name):
        """
        Run a single experiment with given configuration.

        Args:
            config: dict with experiment parameters
            experiment_name: str identifier for this experiment

        Returns:
            dict: Experiment results
        """
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")

        # Create temporary args object
        class Args:
            pass

        args = Args()
        for key, value in config.items():
            setattr(args, key, value)

        # Ensure required attributes
        if not hasattr(args, 'mode'):
            args.mode = 'single'
        if not hasattr(args, 'output_dir'):
            args.output_dir = str(self.output_dir / experiment_name)

        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Run training
            if args.mode == 'single':
                train_single_modality(args)

            # Load results
            metrics_path = Path(args.output_dir) / 'test_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                result = {
                    'experiment_name': experiment_name,
                    'config': config,
                    'metrics': metrics,
                    'success': True
                }
                print(f"\n✓ Experiment complete: AUC={metrics['auc']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}")
            else:
                result = {
                    'experiment_name': experiment_name,
                    'config': config,
                    'success': False,
                    'error': 'Metrics file not found'
                }
                print(f"\n✗ Experiment failed: metrics file not found")

        except Exception as e:
            result = {
                'experiment_name': experiment_name,
                'config': config,
                'success': False,
                'error': str(e)
            }
            print(f"\n✗ Experiment failed: {e}")

        self.results.append(result)
        return result

    def grid_search(self, base_config, param_grid, max_experiments=None):
        """
        Perform grid search over hyperparameters.

        Args:
            base_config: dict with base configuration
            param_grid: dict mapping parameter names to lists of values
            max_experiments: int, maximum number of experiments (None = all)

        Returns:
            list: All experiment results
        """
        print(f"\n{'='*70}")
        print(f"  GRID SEARCH")
        print(f"{'='*70}")
        print(f"Base configuration: {base_config}")
        print(f"Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        print(f"{'='*70}\n")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        if max_experiments and len(combinations) > max_experiments:
            print(f"⚠️  Total combinations ({len(combinations)}) exceeds max_experiments "
                  f"({max_experiments}). Sampling randomly.")
            indices = np.random.choice(len(combinations), max_experiments, replace=False)
            combinations = [combinations[i] for i in indices]

        print(f"Running {len(combinations)} experiments...\n")

        # Run experiments
        for i, param_combo in enumerate(combinations, 1):
            # Create config for this experiment
            config = base_config.copy()
            config.update(dict(zip(param_names, param_combo)))

            # Create experiment name
            param_str = '_'.join([f"{k}={v}" for k, v in zip(param_names, param_combo)])
            experiment_name = f"exp_{i:03d}_{param_str}"

            # Run experiment
            self.run_experiment(config, experiment_name)

        return self.results

    def save_results(self, filename='grid_search_results.json'):
        """Save all results to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self):
        """Print summary of all experiments."""
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")

        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]

        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}\n")

        if successful:
            print("Top 5 experiments by AUC:")
            print(f"{'Rank':<6} {'Experiment':<30} {'AUC':<8} {'Acc':<8} {'F1':<8}")
            print("-" * 70)

            # Sort by AUC
            sorted_results = sorted(
                successful,
                key=lambda x: x['metrics']['auc'],
                reverse=True
            )

            for rank, result in enumerate(sorted_results[:5], 1):
                metrics = result['metrics']
                exp_name = result['experiment_name'][:28]
                print(f"{rank:<6} {exp_name:<30} {metrics['auc']:.4f}   "
                      f"{metrics['accuracy']:.4f}   {metrics['f1']:.4f}")

            print("\n" + "="*70)
            print("Best configuration:")
            best = sorted_results[0]
            print(f"  Experiment: {best['experiment_name']}")
            print(f"  Configuration:")
            for key, value in best['config'].items():
                print(f"    {key}: {value}")
            print(f"  Metrics:")
            for key, value in best['metrics'].items():
                print(f"    {key}: {value:.4f}")

        if failed:
            print(f"\n⚠️  {len(failed)} experiments failed")


def experiment_learning_rate(runner, base_config):
    """Experiment with different learning rates."""
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Learning Rate Sensitivity")
    print("="*70 + "\n")

    param_grid = {
        'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    }

    return runner.grid_search(base_config, param_grid)


def experiment_architecture(runner, base_config):
    """Experiment with different architectural choices."""
    print("\n" + "="*70)
    print("  EXPERIMENT 2: Architecture Variations")
    print("="*70 + "\n")

    param_grid = {
        'hidden_dim': [64, 128, 256],
        'output_dim': [32, 64, 128],
    }

    return runner.grid_search(base_config, param_grid, max_experiments=6)


def experiment_regularization(runner, base_config):
    """Experiment with different regularization strategies."""
    print("\n" + "="*70)
    print("  EXPERIMENT 3: Regularization")
    print("="*70 + "\n")

    param_grid = {
        'weight_decay': [0.0, 0.001, 0.01, 0.1],
        'dropout': [0.0, 0.1, 0.3, 0.5],
    }

    return runner.grid_search(base_config, param_grid, max_experiments=8)


def experiment_training(runner, base_config):
    """Experiment with training hyperparameters."""
    print("\n" + "="*70)
    print("  EXPERIMENT 4: Training Hyperparameters")
    print("="*70 + "\n")

    param_grid = {
        'batch_size': [16, 32, 64],
        'lr': [0.0005, 0.001, 0.002],
    }

    return runner.grid_search(base_config, param_grid)


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter experiments for CogniSense'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='eye',
        choices=['eye', 'typing', 'drawing', 'gait'],
        help='Modality to experiment with'
    )
    parser.add_argument(
        '--experiment-type',
        type=str,
        default='all',
        choices=['all', 'lr', 'arch', 'reg', 'training'],
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=300,
        help='Number of samples (smaller for faster experiments)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Epochs per experiment'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Base configuration
    base_config = {
        'modality': args.modality,
        'num_samples': args.num_samples,
        'epochs': args.epochs,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.01,
        'patience': 5,
        'seed': args.seed,
    }

    # Create runner
    runner = ExperimentRunner(output_dir=args.output_dir)

    # Run experiments
    if args.experiment_type == 'all':
        experiment_learning_rate(runner, base_config)
        experiment_architecture(runner, base_config)
        experiment_regularization(runner, base_config)
        experiment_training(runner, base_config)
    elif args.experiment_type == 'lr':
        experiment_learning_rate(runner, base_config)
    elif args.experiment_type == 'arch':
        experiment_architecture(runner, base_config)
    elif args.experiment_type == 'reg':
        experiment_regularization(runner, base_config)
    elif args.experiment_type == 'training':
        experiment_training(runner, base_config)

    # Save and summarize results
    runner.save_results(
        filename=f'{args.modality}_experiments_{args.experiment_type}.json'
    )
    runner.print_summary()

    print(f"\n{'='*70}")
    print(f"  ✓ All experiments complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

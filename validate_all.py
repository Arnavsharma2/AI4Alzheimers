#!/usr/bin/env python3
"""
Comprehensive Validation Script for CogniSense

Validates all phases (1-5) before proceeding to Phase 6 (PDF Report)
"""

import sys
import os
from pathlib import Path

def validate_phase(phase_num, checks):
    """Run validation checks for a phase"""
    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num} VALIDATION")
    print(f"{'='*60}\n")

    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            status = "✓" if result else "✗"
            print(f"{status} {check_name}")
            all_passed = all_passed and result
        except Exception as e:
            print(f"✗ {check_name}: {e}")
            all_passed = False

    return all_passed

def check_file_exists(path):
    """Check if file exists"""
    return Path(path).exists()

def check_syntax(path):
    """Check Python file syntax"""
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
        return True
    except:
        return False

# PHASE 1 CHECKS
def phase1_checks():
    checks = [
        ("README.md exists", lambda: check_file_exists("README.md")),
        ("requirements.txt exists", lambda: check_file_exists("requirements.txt")),
        ("Demo notebook exists", lambda: check_file_exists("notebooks/CogniSense_Demo.ipynb")),
        ("Gradio demo exists", lambda: check_file_exists("src/demo.py")),
        ("Gradio demo syntax", lambda: check_syntax("src/demo.py")),
        ("Test Phase 1 exists", lambda: check_file_exists("test_phase1.py")),
        ("Test Phase 1 syntax", lambda: check_syntax("test_phase1.py")),
        ("Launch script exists", lambda: check_file_exists("launch_demo.py")),
        ("Launch script syntax", lambda: check_syntax("launch_demo.py")),
    ]
    return validate_phase(1, checks)

# PHASE 2 CHECKS
def phase2_checks():
    checks = [
        ("Training script exists", lambda: check_file_exists("train.py")),
        ("Training script syntax", lambda: check_syntax("train.py")),
        ("Training script executable", lambda: os.access("train.py", os.X_OK)),
        ("Dataset module exists", lambda: check_file_exists("src/data_processing/dataset.py")),
        ("Dataset module syntax", lambda: check_syntax("src/data_processing/dataset.py")),
        ("Training utils exists", lambda: check_file_exists("src/utils/training_utils.py")),
        ("Training utils syntax", lambda: check_syntax("src/utils/training_utils.py")),
        ("TRAINING.md exists", lambda: check_file_exists("TRAINING.md")),
        ("Test Phase 2 exists", lambda: check_file_exists("test_phase2.py")),
        ("Test Phase 2 syntax", lambda: check_syntax("test_phase2.py")),
        (".gitignore exists", lambda: check_file_exists(".gitignore")),
    ]
    return validate_phase(2, checks)

# PHASE 3 CHECKS (Data Processing - mostly skipped, using synthetic)
def phase3_checks():
    checks = [
        ("Synthetic data generator exists", lambda: check_file_exists("src/data_processing/synthetic_data_generator.py")),
        ("Synthetic data generator syntax", lambda: check_syntax("src/data_processing/synthetic_data_generator.py")),
        ("DATASETS.md exists", lambda: check_file_exists("DATASETS.md")),
    ]
    return validate_phase(3, checks)

# PHASE 4 CHECKS
def phase4_checks():
    checks = [
        ("Visualization module exists", lambda: check_file_exists("src/utils/visualization.py")),
        ("Visualization module syntax", lambda: check_syntax("src/utils/visualization.py")),
        ("VISUALIZATION.md exists", lambda: check_file_exists("VISUALIZATION.md")),
        ("Utils __init__ updated", lambda: "visualization" in open("src/utils/__init__.py").read()),
    ]
    return validate_phase(4, checks)

# PHASE 5 CHECKS
def phase5_checks():
    checks = [
        ("Results generation script exists", lambda: check_file_exists("generate_results.py")),
        ("Results generation script syntax", lambda: check_syntax("generate_results.py")),
        ("Results generation executable", lambda: os.access("generate_results.py", os.X_OK)),
        ("RESULTS.md exists", lambda: check_file_exists("RESULTS.md")),
    ]
    return validate_phase(5, checks)

# CORE MODEL CHECKS
def core_models_checks():
    checks = [
        ("Speech model exists", lambda: check_file_exists("src/models/speech_model.py")),
        ("Speech model syntax", lambda: check_syntax("src/models/speech_model.py")),
        ("Eye model exists", lambda: check_file_exists("src/models/eye_model.py")),
        ("Eye model syntax", lambda: check_syntax("src/models/eye_model.py")),
        ("Typing model exists", lambda: check_file_exists("src/models/typing_model.py")),
        ("Typing model syntax", lambda: check_syntax("src/models/typing_model.py")),
        ("Drawing model exists", lambda: check_file_exists("src/models/drawing_model.py")),
        ("Drawing model syntax", lambda: check_syntax("src/models/drawing_model.py")),
        ("Gait model exists", lambda: check_file_exists("src/models/gait_model.py")),
        ("Gait model syntax", lambda: check_syntax("src/models/gait_model.py")),
        ("Fusion model exists", lambda: check_file_exists("src/fusion/fusion_model.py")),
        ("Fusion model syntax", lambda: check_syntax("src/fusion/fusion_model.py")),
    ]
    return validate_phase("CORE", checks)

# STRUCTURE CHECKS
def structure_checks():
    required_dirs = [
        "src/models",
        "src/fusion",
        "src/data_processing",
        "src/utils",
        "notebooks",
        "data/raw",
        "data/processed",
        "models",
    ]

    checks = []
    for dir_path in required_dirs:
        checks.append((f"Directory {dir_path} exists", lambda p=dir_path: Path(p).exists()))

    return validate_phase("STRUCTURE", checks)

# FILE COUNT CHECK
def count_check():
    print(f"\n{'='*60}")
    print(f"  FILE COUNT SUMMARY")
    print(f"{'='*60}\n")

    py_files = list(Path('.').rglob('*.py'))
    py_files = [f for f in py_files if '__pycache__' not in str(f)]

    md_files = list(Path('.').rglob('*.md'))
    md_files = [f for f in md_files if '.git' not in str(f)]

    ipynb_files = list(Path('.').rglob('*.ipynb'))
    ipynb_files = [f for f in ipynb_files if '.ipynb_checkpoints' not in str(f)]

    print(f"Python files: {len(py_files)}")
    print(f"Markdown docs: {len(md_files)}")
    print(f"Notebooks: {len(ipynb_files)}")
    print(f"\nTotal implementation files: {len(py_files) + len(ipynb_files)}")

    return True

def main():
    print("\n" + "="*60)
    print("  CogniSense Comprehensive Validation")
    print("="*60)

    os.chdir('/home/user/AI4Alzheimers')

    results = {}

    # Validate structure first
    results['structure'] = structure_checks()

    # Validate core models
    results['core'] = core_models_checks()

    # Validate each phase
    results['phase1'] = phase1_checks()
    results['phase2'] = phase2_checks()
    results['phase3'] = phase3_checks()
    results['phase4'] = phase4_checks()
    results['phase5'] = phase5_checks()

    # File count
    count_check()

    # Summary
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}\n")

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name.upper()}: {status}")

    all_passed = all(results.values())

    print(f"\n{'='*60}")
    if all_passed:
        print("  ✅ ALL VALIDATIONS PASSED")
        print(f"{'='*60}")
        print("\nProject is ready for Phase 6 (PDF Report)!")
        return 0
    else:
        print("  ❌ SOME VALIDATIONS FAILED")
        print(f"{'='*60}")
        print("\nPlease fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

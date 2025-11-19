#!/usr/bin/env python3
"""
Lightweight code quality and structure validation test.
Tests code without requiring ML dependencies.
"""

import os
import sys
import ast
import py_compile
from pathlib import Path

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\n" + "="*60)
    print("  TEST 1: Python Syntax Validation")
    print("="*60)

    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    errors = []
    for filepath in python_files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"✓ {filepath}")
        except py_compile.PyCompileError as e:
            errors.append((filepath, str(e)))
            print(f"✗ {filepath}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} files with syntax errors")
        return False
    else:
        print(f"\n✅ All {len(python_files)} Python files have valid syntax")
        return True

def test_imports_structure():
    """Test that files have proper import structure."""
    print("\n" + "="*60)
    print("  TEST 2: Import Structure Analysis")
    print("="*60)

    key_files = {
        'src/models/speech_model.py': ['torch', 'transformers'],
        'src/models/eye_model.py': ['torch'],
        'src/models/typing_model.py': ['torch'],
        'src/models/drawing_model.py': ['torch', 'transformers'],
        'src/models/gait_model.py': ['torch'],
        'src/fusion/fusion_model.py': ['torch'],
        'src/data_processing/dataset.py': ['torch'],
        'src/utils/training_utils.py': ['torch', 'sklearn'],
        'train.py': ['torch', 'argparse'],
    }

    all_passed = True
    for filepath, expected_imports in key_files.items():
        if not os.path.exists(filepath):
            print(f"✗ {filepath}: File not found")
            all_passed = False
            continue

        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())

            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            missing = set(expected_imports) - imports
            if missing:
                print(f"✗ {filepath}: Missing imports {missing}")
                all_passed = False
            else:
                print(f"✓ {filepath}: All expected imports present")
        except Exception as e:
            print(f"✗ {filepath}: Error parsing - {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All key files have proper import structure")
    else:
        print("\n❌ Some files have import issues")

    return all_passed

def test_class_definitions():
    """Test that key model classes are defined."""
    print("\n" + "="*60)
    print("  TEST 3: Class Definition Validation")
    print("="*60)

    expected_classes = {
        'src/models/speech_model.py': ['SpeechModel'],
        'src/models/eye_model.py': ['EyeTrackingModel'],
        'src/models/typing_model.py': ['TypingModel'],
        'src/models/drawing_model.py': ['ClockDrawingModel'],
        'src/models/gait_model.py': ['GaitModel'],
        'src/fusion/fusion_model.py': ['MultimodalFusionModel'],
        'src/data_processing/dataset.py': ['MultimodalAlzheimerDataset'],
        'src/data_processing/synthetic_data_generator.py': ['EyeTrackingGenerator', 'TypingDynamicsGenerator', 'ClockDrawingGenerator', 'GaitDataGenerator'],
    }

    all_passed = True
    for filepath, expected in expected_classes.items():
        if not os.path.exists(filepath):
            print(f"✗ {filepath}: File not found")
            all_passed = False
            continue

        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())

            classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

            missing = set(expected) - classes
            if missing:
                print(f"✗ {filepath}: Missing classes {missing}")
                all_passed = False
            else:
                print(f"✓ {filepath}: {', '.join(expected)}")
        except Exception as e:
            print(f"✗ {filepath}: Error parsing - {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All expected classes are defined")
    else:
        print("\n❌ Some classes are missing")

    return all_passed

def test_function_definitions():
    """Test that key functions are defined."""
    print("\n" + "="*60)
    print("  TEST 4: Function Definition Validation")
    print("="*60)

    expected_functions = {
        'src/utils/training_utils.py': ['compute_metrics', 'train_epoch', 'evaluate'],
        'src/utils/visualization.py': ['plot_roc_curve', 'plot_confusion_matrix', 'plot_training_curves'],
        'generate_results.py': ['train_model', 'collect_metrics', 'generate_visualizations'],
    }

    all_passed = True
    for filepath, expected in expected_functions.items():
        if not os.path.exists(filepath):
            print(f"✗ {filepath}: File not found")
            all_passed = False
            continue

        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())

            functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

            missing = set(expected) - functions
            if missing:
                print(f"✗ {filepath}: Missing functions {missing}")
                all_passed = False
            else:
                print(f"✓ {filepath}: {', '.join(expected)}")
        except Exception as e:
            print(f"✗ {filepath}: Error parsing - {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All expected functions are defined")
    else:
        print("\n❌ Some functions are missing")

    return all_passed

def test_docstrings():
    """Test that key files and classes have docstrings."""
    print("\n" + "="*60)
    print("  TEST 5: Docstring Coverage")
    print("="*60)

    key_files = [
        'src/models/speech_model.py',
        'src/models/eye_model.py',
        'src/fusion/fusion_model.py',
        'train.py',
        'generate_results.py',
    ]

    all_passed = True
    for filepath in key_files:
        if not os.path.exists(filepath):
            print(f"✗ {filepath}: File not found")
            all_passed = False
            continue

        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())

            # Check module docstring
            module_doc = ast.get_docstring(tree)
            has_module_doc = module_doc is not None and len(module_doc.strip()) > 0

            # Check class docstrings
            class_docs = 0
            total_classes = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1
                    if ast.get_docstring(node):
                        class_docs += 1

            if has_module_doc and (total_classes == 0 or class_docs == total_classes):
                print(f"✓ {filepath}: Module + {class_docs}/{total_classes} classes documented")
            else:
                print(f"✗ {filepath}: Module doc: {has_module_doc}, Classes: {class_docs}/{total_classes}")
                all_passed = False

        except Exception as e:
            print(f"✗ {filepath}: Error parsing - {e}")
            all_passed = False

    if all_passed:
        print("\n✅ Key files have proper documentation")
    else:
        print("\n⚠️  Some files could use better documentation")

    return all_passed

def test_file_structure():
    """Test that project has expected file structure."""
    print("\n" + "="*60)
    print("  TEST 6: Project Structure")
    print("="*60)

    required_files = [
        'README.md',
        'requirements.txt',
        'train.py',
        'generate_results.py',
        'validate_all.py',
        'src/models/__init__.py',
        'src/fusion/__init__.py',
        'src/data_processing/__init__.py',
        'src/utils/__init__.py',
        'notebooks/CogniSense_Demo.ipynb',
        'report/CogniSense_Report.md',
    ]

    required_dirs = [
        'src/models',
        'src/fusion',
        'src/data_processing',
        'src/utils',
        'notebooks',
        'data',
        'models',
        'report',
    ]

    all_passed = True

    # Check files
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}: Missing")
            all_passed = False

    # Check directories
    for dirpath in required_dirs:
        if os.path.isdir(dirpath):
            print(f"✓ {dirpath}/")
        else:
            print(f"✗ {dirpath}/: Missing")
            all_passed = False

    if all_passed:
        print("\n✅ Project structure is complete")
    else:
        print("\n❌ Some required files/directories are missing")

    return all_passed

def test_requirements():
    """Test that requirements.txt has all necessary packages."""
    print("\n" + "="*60)
    print("  TEST 7: Requirements File")
    print("="*60)

    required_packages = [
        'torch',
        'transformers',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'gradio',
        'numpy',
        'pandas',
    ]

    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False

    with open('requirements.txt', 'r') as f:
        content = f.read().lower()

    all_passed = True
    for package in required_packages:
        if package.lower() in content:
            print(f"✓ {package}")
        else:
            print(f"✗ {package}: Missing from requirements.txt")
            all_passed = False

    if all_passed:
        print("\n✅ All required packages listed in requirements.txt")
    else:
        print("\n❌ Some packages missing from requirements.txt")

    return all_passed

def main():
    """Run all code quality tests."""
    print("\n" + "="*60)
    print("  🧠 CogniSense Code Quality Validation")
    print("="*60)

    tests = [
        ("Syntax", test_python_syntax),
        ("Imports", test_imports_structure),
        ("Classes", test_class_definitions),
        ("Functions", test_function_definitions),
        ("Docstrings", test_docstrings),
        ("Structure", test_file_structure),
        ("Requirements", test_requirements),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("\n" + "="*60)
    if total_passed == total_tests:
        print(f"  ✅ ALL {total_tests} TESTS PASSED")
        print("="*60)
        return 0
    else:
        print(f"  ⚠️  {total_passed}/{total_tests} TESTS PASSED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

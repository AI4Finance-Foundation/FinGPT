#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cloud Testing Script for FinGPT (Python 3.12 Compatibility Check)
This file can be saved as 'cloud_test_fingpt.py' and executed in any 
cloud environment (Google Colab, GitHub Codespaces, Kaggle Kernels) 
to verify that all dependencies and Python 3.12 packages are loaded correctly.
"""

import sys
import importlib.util

def check_package(package_name):
    """Check if a package can be imported and print its version."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"[-] {package_name}: Not Installed")
        return False
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "Version unknown")
        print(f"[+] {package_name}: {version}")
        return True
    except Exception as e:
        print(f"[!] {package_name}: Import failed ({e})")
        return False

def main():
    print("=" * 60)
    print("FinGPT Python 3.12 Cloud Environment Validation Script")
    print("=" * 60)
    
    # 1. Print System & Python Version Information
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version Details:\n{sys.version}")
    print("-" * 60)
    
    # 2. List of core packages required for FinGPT components
    packages_to_test = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "numpy",
        "pandas",
        "yfinance",
        "gradio"
    ]
    
    print("Checking core dependencies:")
    results = {}
    for pkg in packages_to_test:
        results[pkg] = check_package(pkg)
        
    print("-" * 60)
    
    # 3. Check PyTorch CUDA / Hardware Acceleration Availability
    if results.get("torch"):
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Current Device Index: {torch.cuda.current_device()}")
        else:
            print("Note: Running on CPU or non-NVIDIA accelerator environment.")
    
    print("=" * 60)
    print("Validation check completed!")

if __name__ == "__main__":
    main()
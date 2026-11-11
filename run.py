#!/usr/bin/env python3
"""
Simple run script for register classification training
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def run_training():
    """Run the training script"""
    print("Starting training...")
    try:
        subprocess.check_call([sys.executable, "train.py"])
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False
    return True

def main():
    print("=== Register Classification Training ===")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("requirements.txt not found!")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run training
    if not run_training():
        return
    
    print("\n=== Training Complete! ===")
    print("Files created:")
    print("- best_register_model.h5 (best model during training)")
    print("- final_register_model.h5 (final model)")
    print("- label_encoder.pkl (label encoder)")
    print("\nTo make predictions, use:")
    print("python predict.py <image_path>")

if __name__ == "__main__":
    main()
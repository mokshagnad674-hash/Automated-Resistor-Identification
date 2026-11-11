#!/usr/bin/env python3
"""
Training script for Register Classification using EfficientNet-B3
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from model import RegisterClassifier

def load_heic_image(file_path):
    """Load HEIC image files"""
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        
        img = Image.open(file_path)
        img = img.convert('RGB')
        return np.array(img)
    except ImportError:
        print("pillow_heif not installed. Install with: pip install pillow_heif")
        return None
    except Exception as e:
        print(f"Error loading HEIC file: {e}")
        return None

def load_dataset(data_dir, target_size=(300, 300)):
    """Load and preprocess images from directory"""
    images = []
    labels = []
    
    data_path = Path(data_dir)
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"Loading {class_name} images...")
            
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']:
                    try:
                        # Load image
                        if img_file.suffix.lower() == '.heic':
                            img = load_heic_image(str(img_file))
                            if img is None:
                                continue
                        else:
                            img = cv2.imread(str(img_file))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Resize image
                        img = cv2.resize(img, target_size)
                        
                        images.append(img)
                        labels.append(class_name)
                        
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
                        continue
            
            print(f"Loaded {len([l for l in labels if l == class_name])} images for {class_name}")
    
    return np.array(images, dtype=np.float32), np.array(labels)

def main():
    # Configuration
    DATA_DIR = "."  # Current directory containing 10K and 6.8K folders
    EPOCHS = 50
    BATCH_SIZE = 16
    INPUT_SHAPE = (300, 300, 3)
    
    print("Loading dataset...")
    images, labels = load_dataset(DATA_DIR, target_size=INPUT_SHAPE[:2])
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print(f"Total images loaded: {len(images)}")
    print(f"Classes: {np.unique(labels)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    classifier = RegisterClassifier(num_classes=num_classes, input_shape=INPUT_SHAPE)
    model = classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Get callbacks
    callbacks = classifier.get_callbacks('best_register_model.h5')
    
    # Phase 1: Train with frozen base model
    print("\n=== Phase 1: Training with frozen base model ===")
    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning with unfrozen top layers
    print("\n=== Phase 2: Fine-tuning with unfrozen layers ===")
    classifier.unfreeze_top_layers(num_layers=20)
    classifier.compile_model(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    
    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS-20,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Save final model
    model.save('final_register_model.h5')
    print("Training completed! Models saved.")

if __name__ == "__main__":
    main()
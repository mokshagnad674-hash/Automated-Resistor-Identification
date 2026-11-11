#!/usr/bin/env python3
"""
Improved Training script for Register Classification using EfficientNet-B3
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from model import ImprovedRegisterClassifier


def load_dataset(data_dir, target_size=(300, 300)):
    """Load and preprocess images from directory"""
    images = []
    labels = []
    
    data_path = Path(data_dir)
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir() and class_dir.name in ['10K', '6.8K']:
            class_name = class_dir.name
            print(f"Loading {class_name} images...")
            
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() == '.png':
                    try:
                        # Load image (JPG only for consistency)
                        img = cv2.imread(str(img_file))
                        if img is None:
                            continue
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
    # Configuration - Improved settings
    DATA_DIR = "."
    EPOCHS = 50
    BATCH_SIZE = 8  # Smaller batch for better learning
    INPUT_SHAPE = (300, 300, 3)
    
    print("Loading dataset...")
    images, labels = load_dataset(DATA_DIR, target_size=INPUT_SHAPE[:2])
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print(f"Total images loaded: {len(images)}")
    print(f"Classes: {np.unique(labels)}")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        print(f"  {label}: {count} images")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Split data with larger validation set
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, 
        test_size=0.25,  # Larger validation set
        random_state=42, 
        stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create improved model
    classifier = ImprovedRegisterClassifier(num_classes=num_classes, input_shape=INPUT_SHAPE)
    model = classifier.build_model()
    classifier.compile_model(learning_rate=0.0005)  # Lower learning rate
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Improved callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_register_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n=== Phase 1: Training with frozen base model ===")
    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=25,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    classifier.unfreeze_top_layers(num_layers=30)  # More layers
    classifier.compile_model(learning_rate=0.00005)  # Very low learning rate
    
    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS-25,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Test on individual classes
    print("\n=== Per-Class Accuracy ===")
    val_predictions = model.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(val_pred_classes[class_mask] == y_val[class_mask])
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Save final model
    model.save('final_register_model.h5')
    print("Training completed! Models saved.")

if __name__ == "__main__":
    main()
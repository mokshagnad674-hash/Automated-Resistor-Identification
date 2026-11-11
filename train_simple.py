#!/usr/bin/env python3
"""
Simple Training Script
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from simple_model import SimpleRegisterClassifier

def load_dataset(data_dir, target_size=(300, 300)):
    """Load PNG images"""
    images = []
    labels = []
    
    for class_dir in ['10K', '6.8K']:
        class_path = Path(class_dir)
        if not class_path.exists():
            continue
            
        print(f"Loading {class_dir} images...")
        
        for img_file in class_path.glob('*.png'):
            try:
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                
                images.append(img)
                labels.append(class_dir)
                
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
        
        print(f"Loaded {len([l for l in labels if l == class_dir])} images for {class_dir}")
    
    return np.array(images, dtype=np.float32), np.array(labels)

def main():
    print("Loading dataset...")
    images, labels = load_dataset(".")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print(f"Total images: {len(images)}")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        print(f"  {label}: {count} images")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=encoded_labels
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create model
    classifier = SimpleRegisterClassifier(num_classes=2)
    model = classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Train
    print("\\n=== Training ===")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=classifier.get_callbacks('best_register_model.h5'),
        verbose=1
    )
    
    # Fine-tune
    print("\\n=== Fine-tuning ===")
    classifier.unfreeze_top_layers(20)
    classifier.compile_model(learning_rate=0.0001)
    
    model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=classifier.get_callbacks('best_register_model.h5'),
        verbose=1
    )
    
    # Final evaluation
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\\nFinal Accuracy: {val_accuracy:.4f}")
    
    model.save('final_register_model.h5')
    print("Training completed!")

if __name__ == "__main__":
    main()
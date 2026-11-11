#!/usr/bin/env python3
"""
Data Augmentation for Resistor Images
"""

import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image
import pillow_heif

def load_heic_image(file_path):
    """Load HEIC image files"""
    try:
        pillow_heif.register_heif_opener()
        img = Image.open(file_path)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading HEIC file: {e}")
        return None

def augment_image(image, augment_type):
    """Apply specific augmentation to image"""
    h, w = image.shape[:2]
    
    if augment_type == 'rotate_5':
        M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1)
        return cv2.warpAffine(image, M, (w, h))
    
    elif augment_type == 'rotate_-5':
        M = cv2.getRotationMatrix2D((w//2, h//2), -5, 1)
        return cv2.warpAffine(image, M, (w, h))
    
    elif augment_type == 'brightness_up':
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 30)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    elif augment_type == 'brightness_down':
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = cv2.subtract(hsv[:,:,2], 30)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    elif augment_type == 'contrast':
        return cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    
    elif augment_type == 'blur':
        return cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

def augment_dataset():
    """Augment the resistor dataset"""
    base_dir = Path(".")
    augmentations = ['rotate_5', 'rotate_-5', 'brightness_up', 'brightness_down', 'contrast', 'blur']
    
    for class_dir in ['10K', '6.8K']:
        class_path = base_dir / class_dir
        if not class_path.exists():
            continue
            
        print(f"Augmenting {class_dir} images...")
        
        # Create augmented directory
        aug_dir = base_dir / f"{class_dir}_augmented"
        aug_dir.mkdir(exist_ok=True)
        
        for img_file in class_path.glob('*'):
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
                    
                    # Apply augmentations
                    for aug_type in augmentations:
                        aug_img = augment_image(img, aug_type)
                        
                        # Save augmented image
                        output_name = f"{img_file.stem}_{aug_type}.jpg"
                        output_path = aug_dir / output_name
                        
                        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), aug_img_bgr)
                
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        print(f"Augmented {class_dir}: {len(list(aug_dir.glob('*.jpg')))} images created")

if __name__ == "__main__":
    augment_dataset()
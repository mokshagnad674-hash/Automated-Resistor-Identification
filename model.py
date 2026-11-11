#!/usr/bin/env python3
"""
Improved EfficientNet-B3 Register Classification Model with Multi-Input Architecture
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model
import numpy as np

class ImprovedRegisterClassifier:
    def __init__(self, num_classes=2, input_shape=(300, 300, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build improved EfficientNet-B3 with attention and multi-scale features"""
        # Main input
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Enhanced data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.15)(x)
        x = layers.RandomBrightness(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Preprocessing for EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        
        # Load EfficientNet-B3 backbone
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        base_model.trainable = False
        
        # Extract multi-scale features
        # Get intermediate layers for multi-scale feature extraction
        layer_names = [
            'block2a_expand_activation',  # 150x150
            'block4a_expand_activation',  # 75x75  
            'block6a_expand_activation',  # 38x38
            'top_activation'              # 10x10
        ]
        
        feature_layers = [base_model.get_layer(name).output for name in layer_names]
        
        # Multi-scale feature processing
        features = []
        for i, feat in enumerate(feature_layers):
            # Global average pooling for each scale
            pooled = layers.GlobalAveragePooling2D()(feat)
            # Reduce dimensions
            reduced = layers.Dense(128, activation='relu')(pooled)
            features.append(reduced)
        
        # Concatenate multi-scale features
        combined_features = layers.Concatenate()(features)
        
        # Simple classification head without attention
        x = layers.Dense(512, activation='relu')(combined_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Final classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile with improved optimizer and loss"""
        # Use AdamW optimizer for better generalization
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def unfreeze_top_layers(self, num_layers=50):
        """Unfreeze more layers for better fine-tuning"""
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            # Freeze all layers except the top num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
    
    def get_callbacks(self, model_path='best_model.h5'):
        """Enhanced callbacks for better training"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                min_lr=1e-8,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            # Cosine annealing for learning rate
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * 0.95 ** epoch
            )
        ]
        return callbacks
#!/usr/bin/env python3
"""
Simple Working EfficientNet-B3 Model
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model

class SimpleRegisterClassifier:
    def __init__(self, num_classes=2, input_shape=(300, 300, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build simple EfficientNet-B3 model"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        
        # EfficientNet-B3 backbone
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        base_model.trainable = False
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile with simple metrics"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def unfreeze_top_layers(self, num_layers=20):
        """Unfreeze top layers"""
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # Find EfficientNet layer
                layer.trainable = True
                for sublayer in layer.layers[:-num_layers]:
                    sublayer.trainable = False
                break
    
    def get_callbacks(self, model_path='best_model.h5'):
        """Simple callbacks"""
        return [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            )
        ]
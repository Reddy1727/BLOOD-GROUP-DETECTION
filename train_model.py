"""
Quick training script to train a ResNet model on blood group dataset
and save it as model_blood_group_detection_resnet.h5
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Dataset path
dataset_path = 'dataset/dataset_blood_group'

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    print("Please ensure the dataset folder exists with blood group subfolders (A+, A-, AB+, AB-, B+, B-, O+, O-)")
    exit(1)

# Image parameters
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10

print("Loading dataset...")

# Create data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"Classes found: {train_generator.class_indices}")
num_classes = len(train_generator.class_indices)

print("Building ResNet50 model...")

# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom top layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    verbose=1
)

# Save the model
model_path = 'test/model_blood_group_detection_resnet.h5'
os.makedirs('test', exist_ok=True)
model.save(model_path)

print(f"Model trained and saved to {model_path}")
print(f"Training complete! Final accuracy: {history.history['accuracy'][-1]:.4f}")

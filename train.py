import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# PATHS 
train_path = "archive/train"
test_path  = "archive/test"

# LOAD & AUGMENT DATA 
# Training data: augment to add variety
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normalize pixels 0-255 → 0-1
    rotation_range=10,       # randomly rotate up to 10 degrees
    zoom_range=0.1,          # randomly zoom in/out 10%
    horizontal_flip=True,    # randomly flip images left/right
    validation_split=0.15    # use 15% of train data as validation
)

# Test data: only normalize, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),    # resize all images to 48x48
    color_mode="grayscale",  # keep as grayscale (1 channel)
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

# Load validation images
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="validation"
)

# Load test images
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# HANDLE CLASS IMBALANCE
# Calculate class weights so disgust isn't ignored
total = sum([3995, 436, 4097, 7215, 4965, 4830, 3171])
class_weights = {}

counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]
for i, count in enumerate(counts):
    class_weights[i] = total / (7 * count)

# Manually boost underperforming classes
class_weights[2] = class_weights[2] * 1.5  # fear
class_weights[4] = class_weights[4] * 0.6  # reduce neutral dominance
class_weights[5] = class_weights[5] * 1.3  # sad

# BUILD THE CNN 
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Fully connected layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')   # 7 emotions = 7 output neurons
])

model.summary()

# COMPILE 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS 
# Stop training if validation accuracy stops improving
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Save the best model automatically
checkpoint = ModelCheckpoint(
    'emotion_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

#  TRAIN
print("\nStarting training... this will take 20-40 minutes\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=70,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# PLOT RESULTS ON GRAPH
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print("\nTraining complete! Model saved as emotion_model.h5")
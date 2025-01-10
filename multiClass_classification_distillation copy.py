import os
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# Define paths and parameters
DATASET_DIR = "dataset"
OUTPUT_DIR = "data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
MAX_STEPS_PER_EPOCH = 1000

# Create output directories for train, valid, test
subsets = ['train', 'valid', 'test']
waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for subset in subsets:
    for waste_type in waste_types:
        folder = os.path.join(OUTPUT_DIR, subset, waste_type)
        os.makedirs(folder, exist_ok=True)

# Split data into train, valid, and test
random.seed(42)
def split_indices(folder, train_ratio=0.5, valid_ratio=0.25):
    all_files = os.listdir(folder)
    n = len(all_files)
    train_size = int(train_ratio * n)
    valid_size = int(valid_ratio * n)

    random.shuffle(all_files)
    train_files = all_files[:train_size]
    valid_files = all_files[train_size:train_size + valid_size]
    test_files = all_files[train_size + valid_size:]

    return train_files, valid_files, test_files

for waste_type in waste_types:
    source_folder = os.path.join(DATASET_DIR, waste_type)
    train_files, valid_files, test_files = split_indices(source_folder)

    for subset, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        dest_folder = os.path.join(OUTPUT_DIR, subset, waste_type)
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))

# Data generators for training and validation
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

valid_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

# Define the teacher model
teacher_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])

teacher_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the teacher model
teacher_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

teacher_model.save("waste_teacher_model.keras")

# Define the student model
student_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])

# Knowledge distillation loss function
def distillation_loss(y_true, y_pred, teacher_logits, temperature=5):
    soft_targets = tf.nn.softmax(teacher_logits / temperature)
    student_logits = tf.nn.softmax(y_pred / temperature)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(soft_targets, student_logits))
    return loss

# Compile the student model
student_optimizer = tf.keras.optimizers.Adam()
student_model.compile(
    optimizer=student_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the student model with knowledge distillation
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(train_generator):
        if step >= MAX_STEPS_PER_EPOCH:
            print(f"Reached max steps per epoch: {MAX_STEPS_PER_EPOCH}")
            break

        x_batch, y_batch = batch
        teacher_logits = teacher_model(x_batch, training=False)

        with tf.GradientTape() as tape:
            student_logits = student_model(x_batch, training=True)
            loss = distillation_loss(y_batch, student_logits, teacher_logits)

        grads = tape.gradient(loss, student_model.trainable_weights)
        student_optimizer.apply_gradients(zip(grads, student_model.trainable_weights))

        # Add progress output
        if step % 100 == 0:  # Log every 100 steps
            print(f"Step {step}, Loss: {loss.numpy():.4f}")

    # Evaluate on validation data
    val_loss, val_accuracy = student_model.evaluate(valid_generator, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the student model
student_model.save("waste_student_model.keras")

# Evaluate the student model
student_model.evaluate(valid_generator)

# Confusion Matrix and Classification Report for student model
test_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

y_pred = student_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=waste_types, yticklabels=waste_types)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=waste_types))

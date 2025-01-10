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

# Define the model
model = tf.keras.Sequential([
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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

# Evaluate the model
model.save("waste_classifier.h5")

# Confusion Matrix and Classification Report
test_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

y_pred = model.predict(test_generator)
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



print(classification_report(y_true, y_pred_classes, target_names=waste_types))

# Visualize and Predict Individual Test Images
def predict_and_visualize(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = waste_types[np.argmax(prediction)]

    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Test on some random test images
test_images = test_generator.filenames
for i in range(2000):  # Visualize 5 test images
    test_image_path = os.path.join(test_generator.directory, test_images[i])
    predict_and_visualize(test_image_path)

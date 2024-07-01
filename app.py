from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import json
import logging

app = Flask(__name__)
model = None
class_indices = None

def load_model_and_indices():
    global model, class_indices
    if os.path.exists('efficientnet_model.h5'):
        model = load_model('efficientnet_model.h5')
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
    else:
        class_indices = None

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_indices is None:
        return jsonify({'error': 'Model or class indices not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_path = os.path.join('tmp', file.filename)
    file.save(img_path)

    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])

    predicted_class = class_indices[str(class_idx)]
    return jsonify({'class': predicted_class, 'probability': float(prediction[0][class_idx])})

@app.route('/train', methods=['POST'])
def train():
    global class_indices

    # Define paths
    train_dir = request.json.get('train_dir', 'Original_Data/train')
    validation_dir = request.json.get('validation_dir', 'Original_Data/test')
    epochs = request.json.get('epochs', 10)

    # Image Data Generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load Data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    class_indices = train_generator.class_indices

    # Save class indices to a file
    with open('class_indices.json', 'w') as f:
        json.dump({str(v): k for k, v in class_indices.items()}, f)

    # Define the Model
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_indices), activation='softmax')
    ])

    # Compile the Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the Model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs
    )

    # Unfreeze the base model for fine-tuning
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Continue Training
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs
    )

    # Save the Model
    model.save('efficientnet_model.h5')
    load_model_and_indices()

    return jsonify({'message': 'Training completed and model saved'}), 200

if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    load_model_and_indices()
    app.run(host='0.0.0.0', port=5000)

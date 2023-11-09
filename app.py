import os

# Disable GPU support for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load the model
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

model.load_weights('model_weights.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = "uploaded_image.jpg"
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(probabilities)

        # Map class index to label (0 is cat, 1 is dog)
        class_labels = ['Cat', 'Dog']
        predicted_label = class_labels[predicted_class]

        return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)

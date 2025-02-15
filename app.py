from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "D:\Projects\Pneumonia Detector\pnemonia_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = "temp_image.png"
    file.save(file_path)
    
    img_array = preprocess_image(file_path)
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    predicted_label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = float(prediction)
    
    os.remove(file_path)  # Clean up temporary file
    
    return jsonify({'prediction': predicted_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
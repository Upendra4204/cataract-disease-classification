from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model_path = 'model89.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(150, 150)):
    # Convert the PIL image to a NumPy array
    img_array = img_to_array(image)
    # Resize the image to match the model's expected input size
    img_array = tf.image.resize(img_array, target_size)
    # Rescale the pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Make predictions
    predictions = model.predict(img_array)
    # Convert predictions to class label and confidence score
    class_index = (predictions > 0.5).astype("int32")[0][0]
    class_labels = ['Immature', 'Mature']  # Replace with your actual class labels
    predicted_class = class_labels[class_index]
    confidence_score = predictions[0][0]
    return predicted_class, confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make predictions
            image = Image.open(filepath)
            predicted_class, confidence_score = predict_image(image)
            
            # Render the results
            return render_template('results.html', 
                                   predicted_class=predicted_class, 
                                   confidence_score=confidence_score, 
                                   image_file=filename)
    return render_template('index.html')

@app.route('/upload_another', methods=['POST'])
def upload_another():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

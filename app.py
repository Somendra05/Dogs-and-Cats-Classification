from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

# Ensure UTF-8 encoding
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf-8')
else:
    import importlib
    importlib.reload(sys)

app = Flask(__name__)

# Load the pre-trained Keras model
MODEL_PATH = os.path.join('model', 'cats_and_dogs_small.h5')
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']

            # Ensure the uploads directory exists
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'uploads')
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)

            # Save the file to ./uploads with a secure filename
            from werkzeug.utils import secure_filename
            filename = secure_filename(f.filename)
            file_path = os.path.join(upload_path, filename)
            f.save(file_path)
            print(f"File saved to {file_path}")

            # Make prediction
            img = image.load_img(file_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            preds = model.predict(x)
            pred_class = 'Dog' if preds[0] > 0.5 else 'Cat'
            print(f"Prediction: {pred_class}")

            # Convert the prediction result to a UTF-8 encoded string
            return pred_class.encode('utf-8').decode('utf-8')
        except Exception as e:
            print(f"Error during prediction: {e}")
            return str(e).encode('utf-8').decode('utf-8')
    return None

if __name__ == '__main__':
    app.run(debug=True)

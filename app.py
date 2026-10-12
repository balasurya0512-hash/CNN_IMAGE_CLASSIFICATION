import os
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from leaf_disease_cnn import predict_single_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_filename = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No file uploaded.'
        else:
            file = request.files['image']
            if file.filename == '':
                error = 'Please choose an image file.'
            elif not allowed_file(file.filename):
                error = 'Only JPG, PNG, BMP, and WEBP images are allowed.'
            else:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                image_filename = filename

                try:
                    predicted_label, predicted_confidence, _ = predict_single_image(upload_path)
                    prediction = predicted_label
                    confidence = round(float(predicted_confidence) * 100, 2)
                except Exception as exc:
                    error = f'Prediction failed: {exc}'

    return render_template('index.html', prediction=prediction, confidence=confidence, image_filename=image_filename, error=error)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

lesion_type_dict = {
    'nv': 'Melanocytic nevi (Common Mole)',
    'mel': 'Melanoma (Cancerous)',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

lesion_info = {
    'nv': 'Melanocytic nevi are common benign skin lesions.',
    'mel': 'Melanoma is the most dangerous form of skin cancer.',
    'bkl': 'Benign keratosis-like lesions are non-cancerous growths.',
    'bcc': 'Basal cell carcinoma is the most common type of skin cancer.',
    'akiec': 'Actinic keratoses can progress to squamous cell carcinoma.',
    'vasc': 'Vascular lesions are abnormal blood vessel formations.',
    'df': 'Dermatofibroma is a common, non-cancerous skin nodule.'
}

IMG_SIZE = 128
model = None
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def load_trained_model():
    global model
    model_path = 'models/skin_disease_model.keras'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully!")
        return True
    return False

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(img_path):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return None
    
    predictions = model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]
    
    results = []
    for idx in top_indices:
        disease_code = classes[idx]
        confidence = float(predictions[idx])
        results.append({
            'code': disease_code,
            'name': lesion_type_dict.get(disease_code, disease_code),
            'confidence': round(confidence * 100, 2),
            'info': lesion_info.get(disease_code, '')
        })
    
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        img_path = os.path.join(upload_dir, file.filename)
        file.save(img_path)
        
        if model is None:
            if not load_trained_model():
                return jsonify({'error': 'Model not found. Please train the model first.'}), 500
        
        results = predict_disease(img_path)
        
        if results is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        return jsonify({
            'predictions': results,
            'image_path': f'/static/uploads/{file.filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running!'})

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    load_trained_model()
    app.run(debug=False, host='127.0.0.1', port=5000)

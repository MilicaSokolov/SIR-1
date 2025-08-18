import os
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from collections import Counter

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Definiši putanje do PyTorch modela K-fold
MODEL_PATHS = [
    r'C:\Users\dual\Milica\sir 1\app\models\efficientnet_model_stratified_fold_0.pth',
    r'C:\Users\dual\Milica\sir 1\app\models\efficientnet_model_stratified_fold_1.pth',
    r'C:\Users\dual\Milica\sir 1\app\models\efficientnet_model_stratified_fold_2.pth',
    r'C:\Users\dual\Milica\sir 1\app\models\efficientnet_model_stratified_fold_3.pth',
    r'C:\Users\dual\Milica\sir 1\app\models\efficientnet_model_stratified_fold_4.pth'
]

# Učitaj PyTorch K-fold modele u listu
models_for_voting = []
for model_path in MODEL_PATHS:
    model = models.efficientnet_b0(pretrained=False)
    classifier = nn.Sequential(
        nn.Linear(in_features=model.classifier[1].in_features, out_features=256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 5)  # 5 klasa cvetova
    )
    model.classifier = classifier
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)
    model.eval()  # Postavi model u evaluacijski mod
    models_for_voting.append(model)

# Definiši klase cveća
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Preprocesiranje slike za PyTorch modele
def preprocess_image_pytorch(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_tensor = transform(img).unsqueeze(0)  # Dodaj batch dimenziju
    return img_tensor

# Predikcija koristeći ensemble PyTorch K-fold modela sa voting metodom
def test_model_with_voting(models, img_tensor):
    preds_per_model = []

    # Prikupljamo predikcije za svaki model
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            preds_per_model.append(preds)

    # Voting metoda za svaku sliku
    preds_per_sample = np.array(preds_per_model).T
    final_preds = []
    for preds in preds_per_sample:
        vote = Counter(preds).most_common(1)[0][0]  # Majority voting
        final_preds.append(vote)

    return final_preds[0]  # Vrati predviđenu klasu

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            # Sačuvaj učitani fajl
            file_path = os.path.join('static/uploads/', file.filename)
            file.save(file_path)

            # Preprocesiraj sliku za PyTorch modele
            pytorch_img_tensor = preprocess_image_pytorch(file_path)

            # Predikcija pomoću PyTorch K-fold modela
            predicted_class_idx = test_model_with_voting(models_for_voting, pytorch_img_tensor)

            # Vrati predikciju klase
            predicted_class = CLASSES[predicted_class_idx]
            return jsonify({
                'pytorch_class': predicted_class
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import pandas as pd
from torchvision import transforms
import dlib
import cv2
from PIL import Image
import numpy as np
import io
import torch

import sys
sys.path.insert(0, '../desafio')
import model as m
import dataset as d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = m.Face_AntiSpoofing().to(device)
model.load_state_dict(torch.load('../desafio/models/001_16_50.pt'))
model.eval()

def prepare_image(img):
    flag = True
    detector = dlib.cnn_face_detection_model_v1("../desafio/mmod_human_face_detector.dat")
    rect = detector(img, 1)
    for r in rect:
        x1 = r.rect.left()
        y1 = r.rect.top()
        x2 = r.rect.right()
        y2 = r.rect.bottom()
    
    try:
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("static/new_img.jpg", img)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
        img = transform(img)
    except:
        img = np.array([])
        flag = False
        
    return img, flag

def predict_result(img):
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device = device, dtype = torch.float))
    return 'live' if output[0] > 0.5 else 'spoof'

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Tente novamente. A imagem não existe."

        file = request.files['file']
        if not file:
            return

        path = "static/img.jpg"
        file.save(path)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, flag = prepare_image(img)
        
        return render_template('predict.html', label = predict_result(img) if flag else 'vazia')
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True, port = 8890)
from os import walk
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import pandas as pd
seed = 123
rng = np.random.default_rng(seed)

def img_preprocessing(path, detector, upsample = 1, hog = True):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rect = detector(img, upsample)
    if hog:
        for r in rect:
            x1 = r.left()
            y1 = r.top()
            x2 = r.right()
            y2 = r.bottom()
    else:
        for r in rect:
            x1 = r.rect.left()
            y1 = r.rect.top()
            x2 = r.rect.right()
            y2 = r.rect.bottom()
    
    try:
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (224,224))

        return img
    except:
        return np.array([])

    
path_data = './MLChallenge_Dataset/Data'
idx = []
for (dirpath, dirnames, filenames) in walk(path_data):
    idx.extend(dirnames)
    break
    
detector_cnn = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
detector_hog = dlib.get_frontal_face_detector()

data_dict = {
        'id': [],
        'path': [],
        'live': []
    }

for i in tqdm(range(len(idx))):
    path_img = path_data +'/'+idx[i]+'/'

    for (dirpath, dirnames, filenames) in walk(path_img+'live'):
        path = [path_img+'live/'+f for f in filenames]

        for p, f in zip(path, filenames):
            img = img_preprocessing(p, detector_hog)
            if img.shape[0] != 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                path_aux = './img_data/live/'+idx[i]+'_'+f[:-4]+'_live.jpg'
                cv2.imwrite(path_aux, img)

                data_dict['id'].append(idx[i])
                data_dict['path'].append(path_aux)
                data_dict['live'].append(1)
        break
        
    for (dirpath, dirnames, filenames) in walk(path_img+'spoof'):
        path = [path_img+'spoof/'+f for f in filenames]

        for p, f in zip(path, filenames):
            img = img_preprocessing(p, detector_cnn, hog = False)
            if img.shape[0] != 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                path_aux = './img_data/spoof/'+idx[i]+'_'+f[:-4]+'_spoof.jpg'
                cv2.imwrite(path_aux, img)

                data_dict['id'].append(idx[i])
                data_dict['path'].append(path_aux)
                data_dict['live'].append(0)
        break
        
pd.DataFrame.from_dict(data_dict).to_csv('./img_data/data.csv', index = False)
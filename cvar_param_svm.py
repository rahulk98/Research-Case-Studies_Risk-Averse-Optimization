import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

"""
SVM with perturbed images of cats and dogs
lables: 0 -> cats, 1 -> dogs

"""

def svm_obj_func(x, y, w, b, n, lamda):
    margins = np.max(0, 1 - y * (np.dot(w, x.T) + b))
    norm_w = np.linalg.norm(w)**2
    return (1/n) * np.sum(margins) + (lamda / 2) * norm_w

def load_images():
    path = "train/train"
    images = []
    labels = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
            label = 0 if 'cat' in filename else 1
            labels.append(label)
    df = pd.DataFrame({'image': images, 'label': labels})
    return df






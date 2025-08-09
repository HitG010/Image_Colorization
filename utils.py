# utils.py
import numpy as np
from skimage import color

def rgb_to_lab(image):
    lab = color.rgb2lab(image).astype("float32")
    L = lab[:, :, 0:1] / 50.0 - 1.0  
    ab = lab[:, :, 1:] / 128.0       
    return L, ab

def lab_to_rgb(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 128.0
    lab = np.concatenate([L, ab], axis=2)
    rgb = color.lab2rgb(lab.clip(0, 100))
    return rgb

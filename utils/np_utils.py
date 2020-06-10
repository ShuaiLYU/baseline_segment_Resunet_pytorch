from PIL import Image
import numpy as np
import logging
import os
import time


def gray2RGB(img):
    img = np.array(img).squeeze()
    assert img.ndim==2
    return np.tile(img[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8)

def one_hot(input,len):
	one_hot = np.eye(len)[input]

def normalize(x):
    axis = (x.ndim - 2, x.ndim - 1)
    min_value = np.min(x, axis=axis, keepdims=True)
    x = x - min_value
    max_value = np.max(x, axis=axis, keepdims=True)
    x = x / max_value
    return x

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    # relu函数
    return np.maximum(0, x)


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

def transform(image, reverse=False):
	if not reverse:
		image=np.array(image).astype(np.float)
		image=image / 255.0
		return image
	else:
		image = np.array(image)
		image=(image)*255
		image=image.astype("uint8")
		return image



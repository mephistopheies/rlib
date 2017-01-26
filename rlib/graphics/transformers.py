import numpy as np
from lasagne.utils import floatX
from skimage import transform


def img2vgg(img):
    return np.swapaxes(np.swapaxes(floatX(img[:, :, ::-1]), 1, 2), 0, 1)


def vgg2img(data):
    return np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)[:, :, ::-1]


def tanh2img(img):
    return (255*(img + 1)/2).astype(np.uint8)


def img2tanh(img):
    return 2*img/255.0 - 1


def crop_resize(img, out_size=64):
    img = img.copy()
    h, w, _ = img.shape
    if h < w:
        img = transform.resize(img, (out_size, w*out_size/h), preserve_range=True)
    else:
        img = transform.resize(img, (h*out_size/w, out_size), preserve_range=True)
    h, w, _ = img.shape
    img = img[h//2 - int(np.floor(out_size/2.0)):h//2 + int(np.ceil(out_size/2.0)),
              w//2 - int(np.floor(out_size/2.0)):w//2 + int(np.ceil(out_size/2.0))]
    return img.astype(np.uint8)

from PIL import Image
import numpy as np


def read_image(fname, debug=False):
    img_rgb = np.array(Image.open(fname).convert('RGB'))
    if img_rgb.max() <= 1:
        img_rgb = (255*img_rgb).astype(np.uint8)
        if debug:
            print 'color space is fixed'
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    if len(img_rgb.shape) == 2:
        img_rgb = np.dstack([img_rgb, img_rgb, img_rgb])
        if debug:
            print 'colorized'
    if img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]
        if debug:
            print 'alpha removed'
    return img_rgb

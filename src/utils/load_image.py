import numpy as np
from PIL import Image


def load_image(filename, downsample=None):
    target_size = (512, 256)  # (W,H)
    im = Image.open(filename)
    if downsample:
        assert downsample.lower() in ['nearest',
                                      'bilinear'], 'use bilinear downsampling for images and nearest neighbour downsampling for labels'
        if downsample == 'nearest':
            im = im.resize(size=target_size, resample=Image.NEAREST)
        else:
            im = im.resize(size=target_size, resample=Image.BILINEAR)
    return np.asarray(im, dtype='uint8')

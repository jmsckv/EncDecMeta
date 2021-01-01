import numpy as np
from PIL import Image
from typing import Tuple, Union

def load_image(filename, target_size: Tuple=None, downsampling_type: Union['nearest','bilinear']=None):
    """
    Load a png image with X channels, potentially downsample it, and return it as a np.array with X channels.
    In the case of semantic segmentation, usually X=3 for RGB png-images in /data and X=#classes for png-files in /labels. 
    """
    im = Image.open(filename)
    target_size = tuple(int(i) for i in target_size)
    if target_size and target_size != im.size:
        assert downsampling_type in ['nearest','bilinear'], 'If downsampling, you must specify an interpolation type. Must be one of: "nearest","bilinear". You should use bilinear downsampling for images and nearest neighbour downsampling for labels.'
        if downsampling_type == 'nearest':
            im = im.resize(size=target_size, resample=Image.NEAREST)
        else:
            im = im.resize(size=target_size, resample=Image.BILINEAR)
    return np.asarray(im, dtype='uint8')



"""
# Demo
from glob import glob
import os
pth = glob(os.path.join(os.environ['DATAPATH'],'proc','data','train','*'))[0]
print(pth)
img = load_image(pth)
print(img.shape)
img = load_image(pth,(100,200),'nearest')
print(img.shape)
"""
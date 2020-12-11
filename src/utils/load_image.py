import numpy as np
from PIL import Image
from typing import Tuple, Union

def load_image(filename, target_size: Tuple=None, downsampling_type: Union['nearest','bilinear']=None):
    im = Image.open(filename)
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
img = load_image(glob(os.path.join(os.environ['DATAPATH'],'proc','data','train','*'))[0])
print(type(img),img.shape)
"""
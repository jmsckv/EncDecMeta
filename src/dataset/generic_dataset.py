import torch
import torch.utils.data
import numpy as np
import os

from utils.load_image import load_image
from utils.printing import fancy_print
from utils.get_paths import get_paths_for_dataset

class GenericDataset(torch.utils.data.Dataset):
    def __init__(self,fold, config):
        
        self.fold = fold
        self.debug = config.get('debug', 0)
        self.test_run = config.get('test_run', None)
        if config.get('overfit', False): # check metrics to converge when memorizing one sample 
            self.test_run = 1
            self.fold = 'train'
        self.img_dir =  os.path.join(os.environ['DATAPATH'],'data', self.fold)
        self.label_dir = os.path.join(os.environ['DATAPATH'],'labels', self.fold)
        
        # TODO: probably dict comprehension + zip faster        
        self.examples = []
        self.samples_dir = os.listdir(self.img_dir)
        self.n_samples = len(self.samples_dir)
        for fn in self.samples_dir[:self.test_run]:  # important: assumed that file names in  dirs (labels/data) identical
            example = {}
            example["img_path"] = os.path.join(self.img_dir, fn)
            example["label_img_path"] = os.path.join(self.label_dir, fn)
            example["img_id"] = fn
            self.examples.append(example)
        self.num_examples = len(self.examples)


        # get values for channel-wise Z-score normalization, by default we use values computed on Cityscapes dataset (full res)
        self.norm_R_mean =  config.get('norm_R_mean', 0.2868955263625)        
        self.norm_G_mean =  config.get('norm_G_mean', 0.3251330100231946)
        self.norm_B_mean =  config.get('norm_B_mean',  0.2838917598962539)
        self.norm_R_std =  config.get('norm_R_std', 0.1869226144355443)        
        self.norm_G_std =  config.get('norm_G_std', 0.19013295203172015)
        self.norm_B_std =  config.get('norm_B_std',  0.18716106284161413)

        # if we want to downsample, we have to specify 'H' and 'W' different to the actual image resolution in the configuration file
        self.H =   config['H']        
        self.W =   config['W']        

        if self.debug: 
            assert fold in ['train', 'val', 'test']
            assert len(os.listdir(self.img_dir)[:self.test_run]) == len(os.listdir(self.label_dir)[:self.test_run]), 'Image and label directory must contain an equal number of files.'

        if self.debug >1:
            message = f'{str.capitalize(fold)} fold contains {self.n_samples} samples.'
            if self.test_run:
                  message += f' Test run: using only first {self.test_run} samples.'
            fancy_print(message)


    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        label_img_path = example["label_img_path"]

        img = load_image(img_path, (self.H, self.W), 'bilinear')
        label_img = load_image(label_img_path, (self.H, self.W), 'nearest')

        img = img / 255.0
        img = img - np.array([self.norm_R_mean, self.norm_G_mean, self.norm_B_mean])
        img = img /np.array([self.norm_R_std, self.norm_G_std, self.norm_B_std])
        img = np.transpose(img, (2, 0, 1))  # (256, 256, 3) > (3, 256, 256)
        img = img.astype(np.float32)    
        img = torch.from_numpy(img) 
        label_img = torch.from_numpy(label_img).long() 


        return (img, label_img)




"""
d = GenericDataset('train', {'H':1024, 'W': 1024}) # changing spatial resolution
print(d.img_dir)
print(d.label_dir)
i,l = d[2]
print(i.shape,l.shape)
"""
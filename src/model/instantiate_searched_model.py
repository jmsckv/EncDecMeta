import torch.nn as nn
import torch
from copy import deepcopy
import os
import json
import pickle 
import pandas as pd
from model.components.components import get_layer, get_block
from utils.sampling import sample_config


class EncDec(nn.Module):
    """
    Encodes an encoder decoder meta search space abstracted by blocks consisting of layers.
    Each architecture has to consists of at least one downsamling, one encoder, and one upsampling block.
    Each architecture has to consists of an equal number of downsampling and upsampling blocks.
    The architecture is encoded in config['sampled_blocks'][{'D','B','U'}] as a nested list of tuples.
    E.g. config['sampled_blocks']['D']= [[('H',2),('C,1')],[('V',1)],[('V',2)]] would encode 3 downsampling blocks, the first one consisting of 2 layers, the second and third one of 1 layer.
    """

    def __init__(self,  config: dict=None, from_path: str=False):
        super().__init__()
        self.arch_string = []
        self.nn_down = nn.ModuleList()
        self.nn_bottle = nn.ModuleList()
        self.nn_up = nn.ModuleList()
        self.nn_logits = nn.ModuleList()

        if from_path and config is None: # path should point to root directory containing model-related information
            self.deserialize_config(from_path)
            self.arch_string = self.config['arch_string']
            self.load_weights_best_mIoU(from_path)

        else: # sampling
            self.config = sample_config(config)

            if self.config['debug'] >= 1:
                for i in ['D_blocks', 'U_blocks', 'B_blocks']:
                    assert len(self.config[i]) >= 1, "Only networks are allowed with at least one downsamling, bottleneck, and decoder block each." # TODO: drop requirement for bottleneck block
                assert len(self.config['D_blocks']) == len(self.config['U_blocks']), "You must specify an equal number of encoder and decoder blocks."

            exp_f = 2  # expansion factor > double/halve channels in each encoder/decoder block

            # Downsampling blocks
            for i,block in enumerate(self.config['D_blocks']):
                block = deepcopy(block)
                channels_in = int(self.config['base_channels'] *  exp_f ** (i-1) )  # will get ignored in first downsampling block
                channels_out = int(self.config['base_channels'] * exp_f ** i) # will get ignored in second downsampling block
                b,a = get_block('D', i, channels_in, channels_out, block, self.config)
                self.nn_down += [b]
                self.arch_string+= [a]
                if  self.config['debug'] > 1:
                    print(f'Appended this downsampling block to overall network topology: {a,b}:')

            # Bottlenecks blocks
            for i,block in enumerate(self.config['B_blocks']):
                block = deepcopy(block)
                channels_in = channels_out # spatial resolution and number of channels remains unchanged
                b,a = get_block('B', i, channels_in, channels_out, block, self.config)
                self.nn_bottle += [b]
                self.arch_string += [a]
                if  self.config['debug'] > 1:
                    print(f'Appended this bottleneck block to overall network topology: {a,b}:')

            # Upsampling blocks
            for i,block in enumerate(self.config['U_blocks']):
                block = deepcopy(block)
                channels_in = int(channels_out*2) # spatial resolution and number of channels remains unchanged
                channels_out = int(channels_in/4)
                if i == len(self.config['U_blocks']) -1 :
                    channels_out = self.config['base_channels']
                b,a = get_block('U', i, channels_in, channels_out, block, self.config)
                self.nn_up += [b]
                self.arch_string += [a]
                if  self.config['debug'] > 1:
                    print(f'Appended this upsampling to overall network topology: {a,b}:')

            # here we could place an upsampling operation if we take an downsampled input but want to predict full resolution

            # last block is currently fixed: out-conv, returning logits
            self.nn_logits += [get_layer('C', self.config['base_channels'], self.config['classes'], self.config, out_conv=True)]

            # turn arch string list into arch string: blocks are seperated by '*'
            self.arch_string = '*'.join(self.arch_string)
            self.config['arch_string'] = self.arch_string 
        
            if  self.config['debug'] > 1:
                print(f'Training this architecture: {self.arch_string}')

    


    def forward(self, x):

        if self.config['debug'] >= 1:
            for i in [2,3]:
                assert x.shape[i] % 2 == 0, 'Input resolution must be even before every downsampling block.'

        tmp_res = [] # store intermediary results of encoder

        # encoder
        for i,b in enumerate(self.nn_down):
            if self.config['debug'] > 1: print(f'Downsampling block number {i}, ingoing tensor shape {x.shape}')
            x = b(x)
            tmp_res += [x]
            if self.config['debug'] > 1: print(f'Downsampling block number {i}, outgoing tensor shape {x.shape}')
            
            if self.config['debug'] >= 1:
                if i < len(self.nn_down)-1: 
                    for j in [2,3]:
                        assert x.shape[j] % 2 == 0, 'Input resolution must be even before every downsampling block.'


        # bottleneck
        for i,b in enumerate(self.nn_bottle):
            if self.config['debug'] > 1:
                 print(f'Bottleneck block number {i}, ingoing tensor shape {x.shape}')
            x = b(x)
            if self.config['debug'] > 1:
                 print(f'Bottleneck block number {i}, outgoing tensor shape {x.shape}')
            
   
        # decoder
        for i,b in enumerate(self.nn_up):
            if self.config['debug'] > 1: 
                print(f'Upsampling block number {i}, ingoing tensor shape {torch.cat([x, tmp_res[::-1][i]], dim=1).shape}')
            x = b(torch.cat([x, tmp_res[::-1][i]], dim=1))
            if self.config['debug'] > 1: 
                print(f'Upsampling block number {i}, outgoing tensor shape {x.shape}')


        logits = self.nn_logits[0](x)

        if self.config['debug'] > 1 : 
            print(logits.shape)

        return logits

    
    def serialize_config(self, path):
        fn = os.path.join(path,'config.pickle')
        with open(fn,'wb') as wf:
            pickle.dump(self.config,wf)
        fn = os.path.join(path,'config.txt')
        with open(fn,'w') as wf:
            for k in sorted(self.config.keys()):
                wf.writelines('\n' + k + ' : ' +  str(self.config[k]))

        

    def deserialize_config(self, path):
        fn = os.path.join(path,'config.pickle')
        with open (fn,'rb') as rf:
            self.config = pickle.load(rf)
    
    def load_weights_best_mIoU(self, path):
        """Load the weights of the best mIoU observed for any model."""
        df = pd.read_csv(os.path.join(path,'progress.csv'))['best_mIoU_val']
        first_epoch =  df[df == df.max()].index.min()
        fn = os.path.join(path,'checkpoint_'+ str(first_epoch), 'model.pt')
        # TODO: make model (de)serialization properly working: https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        # self.load_state_dict(torch.load(fn))
        self.network = torch.load(fn)
        self.network.eval()




























""" 
# Demo 1 

cfg = {}
cfg['experiment_name'] = 'test123'
cfg['model'] = {}
cfg['dropout_ratio'] = [0.1, 0.2]
cfg['momentum_bn'] = (0.1,0.2)
cfg['momentum'] = 0.5
cfg['lr'] = (0.001, 0.1)
cfg['H'] = 1024
cfg['W'] = 1024
cfg['base_channels'] = 8
cfg['channels'] = 3
cfg['classes'] = 7
cfg['debug'] = 2
cfg['batch_size'] = 1
# note: usually padding_mode and activation_function do not have to be specified, they are autofilled when validating the configuration file previous to running sample_and_train.py
cfg['padding_mode'] = 'replicate'
cfg['activation_function'] = nn.ReLU()

# arch-specification syntax
# operation in 'HCVO'
# int(dilation) >=1
# layer = (operation,dilation)
# block = [layer,layer,layer]
# D/B/U = [block,block,block]
complicated_block = [('H',3),('V',3),('C',4)] * 2
cfg['D_blocks'] = [[],[],complicated_block] #* 3
cfg['B_blocks'] = [[('C',6)],[('H',3)]]
cfg['U_blocks'] = [[('H',2)],[],[('V',4)]] # * 3
e = EncDec(cfg)
t = torch.ones(size=[1, 3, 256*4, 512*4]) # *4 * 4
o = e(t)
print(o.shape)


# Demo 2 -  illustrate that we are actually sampling
# we'll sample the bottleneck blocks
cfg = {}
cfg['experiment_name'] = 'test123'
cfg['model'] = {}
cfg['dropout_ratio'] = [0.1, 0.2]
cfg['momentum_bn'] = (0.1,0.2)
cfg['momentum'] = 0.5
cfg['lr'] = (0.001, 0.1)
cfg['H'] = 1024
cfg['W'] = 1024
cfg['base_channels'] = 8
cfg['channels'] = 3
cfg['classes'] = 7
cfg['debug'] = 2
# note: usually padding_mode and activation_function do not have to be specified, they are autofilled when validating the configuration file previous to running sample_and_train.py
cfg['padding_mode'] = 'replicate'
cfg['activation_function'] = nn.ReLU()
cfg['batch_size'] = 10
cfg['D_blocks'] = [[],[]]
cfg['U_blocks'] = [[],[]]
sampled_b_layer = (['H','V','O','C'],range(1,10))
cfg['B_blocks'] = [[sampled_b_layer]*2]*2 # 2 bottleneck blocks with 2 layers each will get sampled


for i in range(5):
    print(f'\nIteration: {i}')
    e = EncDec(cfg)
    print(f'\nSampled arch: {e.arch_string}')
    print(f'\nSampled bottleneck blocks: {e.config["B_blocks"]}')
    for j in ['lr', 'momentum','momentum_bn','dropout_ratio']: 
        print(f"Sampled {j}: {e.config[j]}")


# Demo 3 - Chargrid arch
# deviating from base channel allocation
c1 = ('C', 1)
c2 = ('C', 2)
c4 = ('C', 4)
c8 = ('C', 8)
cfg['D_blocks'] = [[c1,c1],[c1,c1],[c2,c2]] 
cfg['B_blocks'] = [[c4,c4,c4],[c8,c8,c8]]
cfg['U_blocks'] =  [[c1,c1],[c1,c1],[c1,c1]] 
#e = EncDec(cfg)
#t = torch.ones(size=[1, 3, 256*4, 512*4]) # *4 * 4
#o = e(t)
#print(o.shape)
"""

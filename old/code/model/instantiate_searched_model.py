# get and set paths
import os
import sys
sys.path.extend(os.environ['PYTHONPATH'])

import torch.nn as nn
import torch
from model.components.components import sample_block_and_return_nn, get_layer
from utils.sampling import sample_flat_hp

class ChargridSearched(nn.Module):
    """
    Encodes a block model space derived from the proposed architecture in Chargrid paper.
    Chargrid architecture can be interpreted as one sampled arch from this model space.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.arch_string = []
        self.nn = nn.ModuleList()

        # abbreviations to make code below cleaner
        I = self.config['dataset'][self.config['dataset']['selected']]['input_channels']
        B = self.config['model']['arch']['n_base_channels']
        C = self.config['dataset'][self.config['dataset']['selected']]['n_classes']

        # Block 1-8
        # 1-3: encoder + downsampling
        # 4-5: encoder
        # 6-8: decoder = encoder + upsampling

        exp_f = 1  # expansion factor of base channels

        for bl in range(1,9): # append one block after another to the module list
            max_l = 4 # number of layers per block: in encoder and middle blocks 3,
            if bl >= 6: # in decoder blocks 4
                max_l += 1

            # sample layers for a given block
            c, d = [], []
            for i in range(1, max_l): # append one layer after another to a block
                c += [sample_flat_hp(self.config, 'b' + str(bl) + '_l' + str(i) + '_op', arch_hp=True)]
                d += [sample_flat_hp(self.config, 'b' + str(bl) + '_l' + str(i) + '_dil', arch_hp=True)]
            # turn these layers into a valid nn.Sequential and create a string representation of the sampled arch
            if bl == 1:
                b, a = sample_block_and_return_nn(I, B, c, d, config=self.config)
            elif bl in [2,3,4]:
                  b, a = sample_block_and_return_nn(B*exp_f, B*exp_f*2, c, d, config=self.config)
                  exp_f *= 2
            elif bl == 5:
                b, a = sample_block_and_return_nn(B * exp_f, B * exp_f , c, d,config=self.config) # exp_f = 8
            elif bl in [6,7,8]:
                b, a = sample_block_and_return_nn(B * exp_f + int(exp_f/2) * B  , int(exp_f/2) * B, c, d,config=self.config)
                exp_f = int(exp_f/2)
            if config['debug']:
                print(f'Sampled block {bl, a}:')
                print(b)
            self.nn.append(b)
            self.arch_string.append(a)

        # Block 9
        # out-conv, returning logits, currently not searched
        self.nn.append(get_layer('C', B, C, self.config, out_conv=True))
        self.arch_string += ['C']

        # turn arch string list into arch string: blocks are seperated by '*', update model name
        self.config['arch_string'] = '--'.join(self.arch_string)
        if self.config['replicate_chargrid']:
             assert self.config['arch_string'] ==  "D|C|C--D|C|C--D|C2|C2--C4|C4|C4--C8|C8|C8--O|T|C|C--O|T|C|C--O|T|C|C--C"

    def forward(self, x):

        # encoding

        #if self.config['debug']: print(x.shape)

        x1 = self.nn[0](x)
        #if self.config['debug']: print(x1.shape)

        x2 = self.nn[1](x1)
        #if self.config['debug']: print(x2.shape)

        x3 = self.nn[2](x2)
        #if self.config['debug']: print(x3.shape)

        x4 = self.nn[3](x3)
        #if self.config['debug']: print(x4.shape)

        x5 = self.nn[4](x4)
        #if self.config['debug']: print(x5.shape)

        # decoding

        x6 = self.nn[5](torch.cat([x3, x5], dim=1))
        #if self.config['debug']: print(x6.shape)

        x7 = self.nn[6](torch.cat([x2, x6], dim=1))
        #if self.config['debug']: print(x7.shape)

        x8 = self.nn[7](torch.cat([x1, x7], dim=1))
        #if self.config['debug']: print(x8.shape)

        logits = self.nn[8](x8)
        #if self.config['debug']: print(logits.shape)

        return logits


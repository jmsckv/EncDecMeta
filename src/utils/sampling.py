import torch
import torch.nn as nn
import random
from copy import deepcopy
import warnings
from typing import Union, List, Dict, Tuple, Any


# TODO: consider validatting with mypy: http://mypy-lang.org/

def test_import():
    print('test_import ABC')


def sample_randomly(choice:Any):
    """Sample hyperparameters and architecture."""
    if isinstance(choice,(range,list)):
        return random.choice(choice)
    elif isinstance(choice,tuple):
        assert len(choice) == 2, 'Tuple should contain exactly 2 entries.'
        for i in [0,1]:
            assert isinstance(choice[i],(int,float)), 'Tuple entries should be specified as integer or floats.'
        return random.uniform(choice[0],choice[1])
    else: 
        return choice


def sample_block(b:List[Tuple[str,int]]):
    """
    Validate and sample a block specified in the config file.
    """

    # validate
    assert isinstance(b,list), 'A block must be specidied as a list of tuples.'
    if len(b):
        for l in b:
            assert isinstance(l,tuple), 'A block must be specidied as a list of tuples.'
            if isinstance(l[0],list):
                for ll in l[0]:
                    assert isinstance(ll,str), 'If you specify the first tuple entry as a list (i.e. you want to sample), then every entry in this list must be a string.'
                    assert ll in 'OHVC', f'Undefined operations {ll}; operation should be one of O,H,V,C.'
            else:
                assert isinstance(l[0],str) , 'If not sampling, the first entry in every layer must be one of: H,V,C,O.' 
                assert l[0] in 'OHVC', f'Undefined operations {l[0]}; operation should be one of O,H,V,C.'

            if isinstance(l[1],(list,range)):
                for ll in l[1]:
                    assert isinstance(ll,int), 'If you specify the second tuple entry as a list (i.e. you want to sample), then every entry in this list must be an integer.'
            else:         
                assert isinstance(l[1],int), 'The second entry in every layer must be of type int' 
        # sample
    block_out = []
    for l in b:
        if len(b):
            block_out.append((sample_randomly(l[0]),sample_randomly(l[1])))
        else:
            block_out.append([])
    return block_out



def sample_blocks(blocks:List[List[Tuple[str,int]]]):
    """
    Validate all blocks of a network part (downsampling/bottleneck/upsampling).
    """
    blocks_out = []
    for b in blocks:
        blocks_out.append(sample_block(b))
    return blocks_out





def sample_arch(config):
    """
    Sample and validate architecture/ search space specified in config file.
    """

    # validate 
    for k in ['D_blocks','B_blocks','U_blocks']:
        assert k in config.keys(), f'Configuration file missing entry for {k}.'
        assert len(config[k]) >= 1, f'You need to specify at least one block per network part. Requirement not satisfied for {k}.'

    assert len(config['D_blocks']) == len(config['U_blocks']), 'Meta search space expects equal number of downsampling and upsampling blocks.'

    for b in config['B_blocks']:
        for l in b:
            assert len(l),  'In bottleneck blocks you need to specify at least one layer per block.'


    # sample
    sampled_arch = {}
    for k in ['D_blocks','B_blocks','U_blocks']:
        sampled_arch[k] = sample_blocks(config[k])
    return sampled_arch





def validate_and_sample_config(config:dict):

    sampled = dict()
    sampled['backup_orig'] = deepcopy(config)


    for k in ['dropout_ratio','momentum_bn','momentum','learning_rate', 'H','W']:
        assert k in config.keys(), f'You forgot to specify mandatory hyperparameter {k}.'
        sampled[k] = sample_randomly(config[k])


    sampled.update(sample_arch(config))

    # first, we aid the user in specifying the search space
    # one mistake that may easily happen is to specify too many downsampling blocks resulting in an odd resolution
    # this would brake the modular block design, where the resoulution gets halved/doubled in every block

    n_down = len(sampled['D_blocks'])
    for r in 'HW':
        for i in range(n_down):
            assert sampled[r]%(2**(i+1)) == 0, f"The {r} resolution becomes uneven after downsampling {i+1} times. Consider using less downsampling blocks or adjusting the input resolution."


    # another mistake that may happen is choosing padding mode == 'circular', 'reflect' while downsampling too much and selecting a too high dilation rate
    # for padding mode == 'zeros' and for 'repeat' this is prevented by the default since we  always pad as many pixel as the selected dilation rate
    # a user can easily still try these adding mode == 'circular', 'reflect', and will get thrown an error in the forward pass of the network
    #  to do so, comment out the line below, and specify the config file accordingly
    sampled['padding_mode'] = 'repeat'


    # the activation function could also be easily sampled, no interaction effects expected
    sampled['activation_function'] = nn.ReLU()



    return sampled






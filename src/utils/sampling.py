import torch
import torch.nn as nn
import random
from copy import deepcopy
import warnings
from typing import Union, List, Dict, Tuple, Any
from datetime import datetime
from datetime import datetime
from uuid import uuid4


def sample_randomly(choice:Any):
    """Sample hyperparameters and architecture."""
    if isinstance(choice,(range,list)):
        return random.choice(choice)
    elif isinstance(choice,tuple):
        return random.uniform(choice[0],choice[1])
    else: 
        return choice

def sample_block(b:List[Tuple[str,int]]):
    """Sample a block within one of the three network parts: downsampling, bottleneck, or upsampling."""
    block_out = []
    for l in b:
        if len(b):
            block_out.append((sample_randomly(l[0]),sample_randomly(l[1])))
        else:
            block_out.append([])
    return block_out


def sample_blocks(blocks:List[List[Tuple[str,int]]]):
    """
    Sample all blocks of a network part (downsampling/bottleneck/upsampling).
    """
    blocks_out = []
    for b in blocks:
        blocks_out.append(sample_block(b))
    return blocks_out


def sample_arch(config):
    """Sample an architecture specified in a config file."""
    sampled_arch = {}
    for k in ['D_blocks','B_blocks','U_blocks']:
        sampled_arch[k] = sample_blocks(config[k])
    return sampled_arch



def validate_config(config:dict):
    """
    Validate a configuration file. 
    This happens before any model gets sampled/instantiated.
    Additionally, configuration file is enriched with default values in case no of missing entries.
    Note: there is an overlap with configuration['debug'] >=1.
    However, this function here checks once upfront the model instantiation, whereas the debug flag performs sanity checks throughout the training process.
    """

    assert isinstance(config,dict), 'The configuration file must be specified as a dictionary.'

    # validate presence of keys
    assert isinstance(config['experiment_name'], str), "Please name the experiment you're going to run. Provide a string. The string will be expanded by a datetime."
    for i in 'HW':
        assert isinstance(config[i], int), f"Please specify the input resolution of any sampled network, you forgot to specify {i}. \
         The input resolution may vary from the actual image resolution, in this case the image and the labels will get size-adjusted accordingly."
    for k in ['dropout_ratio','momentum_bn','momentum','lr', 'H','W', 'weight_decay', 'nesterov']:
        assert k in config.keys(), f'You forgot to specify the mandatory hyperparameter {k}.'

    # validate keys specified as tuples: these are intended to  model sampling from a continuous uniform distribution
    for k in config:
        if isinstance(k,tuple):
            assert len(k) == 2, 'Tuple should contain exactly 2 entries.'
            for i in [0,1]:
                assert isinstance(k[i],(int,float)), f'Tuple entries should be specified as integer or floats. Requirement not satifsfied for {k}.'

    # validate arch-related keys 
    for k in ['D_blocks','B_blocks','U_blocks']:
        assert k in config.keys(), f'Configuration file missing entry for {k}.'
        assert len(config[k]) >= 1, f'You need to specify at least one block per network part. Requirement not satisfied for {k}.'
    assert len(config['D_blocks']) == len(config['U_blocks']), 'Meta search space expects equal number of downsampling and upsampling blocks.'
    # another mistake that may easily happen is to specify too many downsampling blocks resulting in an odd resolution
    # this would brake the modular block design, where the resoulution gets halved/doubled in every block
    n_down = len(config['D_blocks'])
    for r in 'HW':
        for i in range(n_down):
            assert config[r]%(2**(i+1)) == 0, f"The {r} resolution becomes uneven after downsampling {i+1} times. Consider using less downsampling blocks or adjusting the input resolution."
    

    # validate specification of architecture/search space

    # this is a high-level summary of the 'grammar' according to which we specifiy the config
    # a fixed architecture is encoded as follows:
    # operation in 'HCVO'
    # int(dilation) >=1
    # layer = (operation,dilation)
    # block = [layer,layer,layer,...]
    # D/B/U = [block,block,block,...]

    for k in ['D_blocks','B_blocks','U_blocks']:
        assert isinstance(config[k],list), f'Each network part has to be specified as a (nested) list. Requirement not satified for {k}.'
        for b in config[k]:
            assert isinstance(b,list), f'A block must be specidied as a list of tuples.Requirement not satisfied for {k,b}'
            if len(b):
                for l in b:
                    assert isinstance(l,tuple), f'A block must be specidied as a list of tuples. Requirement not satisified for {k,b,l,l}'
                    assert len(l) == 2, f'A block must be specidied as a list of tuples. Each tuple must contain exactly two entries. Requirement not satisified for {k,b,l}'
                    if isinstance(l[0],list):
                        for ll in l[0]:
                            assert isinstance(ll,str), f'If you specify the first tuple entry as a list (i.e. you want to sample), then every entry in this list must be a string. Requirement not satisified for {k,b,l,ll}'
                            assert ll in 'OHVC', f'Undefined operations {ll}; operation should be one of O,H,V,C. Requirement not satisified for {k,b,l,ll}'
                    else:
                         assert l[0] in 'OHVC', f'If not sampling from a list, the first entry in every layer must be one of: H,V,C,O. Requirement not satisified for {k,b,l}'

                    if isinstance(l[1],(list,range)):
                        for ll in l[1]:
                            assert isinstance(ll,int), f'If you specify the second tuple entry as a list or range object (i.e. you want to sample), then every entry in this list must be an integer. Requirement not satisified for {k,b,l,ll}.'
                            assert ll >= 1, f'You must sample a dilation rate of at least 1. Requirement not satisified for {k,b,l,ll}.'

                    else:         
                        assert isinstance(l[1],int), f'If not a list or range object, the second entry in every layer must be of type int. Requirement not satisified for {k,b,l}'
                        assert l[1] >= 1, f'You must sample a dilation rate of at least 1. Requirement not satisified for {k,b,l}.'

    # fill missing required vales with default ones
    config['num_samples'] = config.get('num_samples', 1)
    config['debug'] = config.get('debug', 0)
    config['channels'] = config.get('channels', 3)
    config['classes'] = config.get('classes', 20) 
    config['key_metric'] = config.get('key_metric','mIoU_val')
    # a mistake that may happen is choosing padding mode == 'circular', 'reflect' while downsampling too much and selecting a too high dilation rate
    # for padding mode == 'zeros' and for 'repeat' this is prevented by the default since we  always pad as many pixel as the selected dilation rate
    # a user can easily still try these adding mode == 'circular', 'reflect', and will get thrown an error in the forward pass of the network
    config['padding_mode'] = config.get('padding_mode', 'replicate')
    # the activation function could also be easily sampled, no interaction effects expected
    config['activation_function'] = config.get('activation_function', nn.ReLU()) 

    return config


def sample_config(config:dict):
    """
    Sample a single model from a configuration file. This configuration file should have been previously validated by validate_config().
    """
    sampled = dict()
    sampled['backup_orig'] = deepcopy(config)
    sampled['experiment_name'] = config['experiment_name'] + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-') + str(uuid4())

    for k in config:
        if k not in ['U_blocks','D_blocks','B_blocks']: 
            sampled[k] = sample_randomly(config[k])

    sampled.update(sample_arch(config))

    return sampled
    
    

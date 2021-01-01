import unittest
import itertools
from copy import deepcopy
from typing import List, Any, Dict
import random
import os
import torch
import torch.nn as nn


class TestGenerator():
    """
    Generates a Cartesian product of test cases. 
    Fixed hyperparameters/settings to be specified in first init-dictionary (flat, not in a container), variable ones as lists in second dictionary.
    Example: 
    f,v = {f:'fixed setting'}, {'a':['A', 'AA', 'AAA'],'b': ['B','BB']}
    tg = TestGenerator(f,v)
    for i in tg.generate_tests: 
        print(i)
    Print-out:
    {'f': 'fixed setting', 'a': 'A', 'b': 'B'}
    {'f': 'fixed setting', 'a': 'A', 'b': 'BB'}
    {'f': 'fixed setting', 'a': 'AA', 'b': 'B'}
    {'f': 'fixed setting', 'a': 'AA', 'b': 'BB'}
    {'f': 'fixed setting', 'a': 'AAA', 'b': 'B'}
    {'f': 'fixed setting', 'a': 'AAA', 'b': 'BB'}
    """
    
    def __init__(self, cfg_fixed:Dict[str,List[Any]]={}, cfg_var:Dict[str,List[Any]]={}, quicktest:bool=False):
        self.cfg_fixed = cfg_fixed
        self.cfg_var = cfg_var
        self.quicktest = quicktest
        if 'QUICKTEST' in os.environ:
            self.quicktest = eval(os.environ['QUICKTEST'])
            assert isinstance(self.quicktest, bool), 'If you specify quicktest as an environment variable, it must be either True or False.'

    @property
    def generate_tests(self):
        if self.quicktest:
            self.cfg_var = {k:[self.cfg_var[k][random.randint(0,len(self.cfg_var[k])-1)]] for k in self.cfg_var}
        for i in (dict(zip(self.cfg_var, x)) for x in itertools.product(*self.cfg_var.values())):
            yield {**self.cfg_fixed, **i}


"""
# Demo: generate Cartesian of all test configurations
F,V = {'f':'fixed setting'}, {'a':['A', 'AA', 'AAA'],'b': ['B','BB']}
tg = TestGenerator(F,V,quicktest=False) # False by default
for i in tg.generate_tests: 
        print(i)
""" 

"""
# Demo: quicktest with single randomly sampled configurations
F,V = {'f':'fixed setting'}, {'a':['A', 'AA', 'AAA'],'b': ['B','BB']}
for i in range(5): 
    tg = TestGenerator(F,V,quicktest=True)
    for j in tg.generate_tests: 
        print(j)
"""

"""
# Demo: how to use only  fixed test cases, i.e. don't create a Cartesian product
tg.cfg_var = {} 
for i in tg.generate_tests: 
    print(i)
"""

""" 
# Demo: extending test generator with new test variable
F,V = {'f':'fixed setting'}, {'a':['A', 'AA', 'AAA'],'b': ['B','BB']}
tg = TestGenerator(F,V,quicktest=False) # False by default
tg.cfg_var['new_var'] = [1,2,3]
#i = next(tg.generate_tests)
for i in tg.generate_tests: 
        print(i)
"""



class EncDecTestGenerator(TestGenerator):
    """
    A prepolulated test generator for our purpose, so that we don't have to remember which arguments to specify in cfg_fixed/cfg_var.
    We can simply update the entries of self.cfg_var for particular test cases.
    """
    def __init__(self, quicktest=False):
        super().__init__(quicktest=quicktest,
                        cfg_fixed=
                                {'block_types':'DBU',
                                'operation_types':'DUOCHV',
                                'debug':1,  # 0 = deactivated, 1 = assertions-checks , 2 = assertion-checks & print-outs
                                'dropout_ratio':0.1,
                                'momentum_bn': .1,
                                'padding_mode': 'replicate', # also supported: zeros; not supported by default: replicate', 'circular' > would require to insure that min. input resolution < 2*padding=dilation since we guarantee that spatial resolution is maintained with padding=dilation
                                'activation_function': nn.ReLU(),
                                'channels':3,
                                'base_channels':64,
                                'classes':20,
                                'H': 256, 
                                'W': 512,
                                'batch_size': 10,
                                'number_of_layers': 3,
                                'dilation': 4,
                                'lr': 0.01,
                                'momentum': 0.1,
                                'num_samples':1,
                                'D_blocks':[[],[]],
                                'U_blocks':[[],[]],
                                'B_blocks':[[('H',3)]],
                                'experiment_name':'encdectg'})
                         #cfg_var={
                                #'activation_function': [nn.ReLU(), nn.LeakyReLU(negative_slope=0.05)]
                                #'padding_mode': ['replicate'],  
                                #'channels':[1,3,10],
                                #'base_channels':[1,63], 
                                #'classes':[1,20],
                                #'H': [16,64,2048], 
                                #'W': [16,512],
                                #'batch_size': [1,5],
                                #'number_of_layers':[0,3],
                                #'dilation': [1,3,1000]})

    @staticmethod
    def get_input_tensor(d):
        return torch.ones(size=(d['batch_size'],d['channels'],d['H'],d['W']))
    
    @staticmethod
    def get_output_tensor(d):
        return torch.ones(size=(d['batch_size'],d['classes'],d['H'],d['W']))
    
    @staticmethod
    def get_ingoing_tensor(d):
        return torch.ones(size=(d['batch_size'],d['in_channels'],d['H'],d['W']))

    @staticmethod
    def get_outgoing_tensor(d):
        return torch.ones(size=(d['batch_size'],d['out_channels']*2,d['H'],d['W']))


    @staticmethod
    def get_output_tensor(d):
        return torch.ones(size=(d['classes'],d['batch_size'],d['H'],d['W']))


"""
tg = EncDecTestGenerator()
i = 0
for j in tg.generate_tests:
    i += 1
print(f'Currently {i} base test cases.') # 1
"""


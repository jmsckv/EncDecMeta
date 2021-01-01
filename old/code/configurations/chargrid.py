"""
This is the base config for the architecture and hyperparameters (HPs) as in Chargrid paper.

General notes on how any config should be constructed:

Config is a nested dictionary.
It's composed of 3  parts:
- config['tune']: everything related to model of arch and HPs.
- config['dataset']: dataset specific configurations such as number of epochs, input resolution, etc.
- other kwargs: directly stored in config[<kwarg>]

Most important config[<kwarg>] to be set is config['fixed_arch'].
If set to True, this indicates that we are not performing a search, but intend to train a single architecture.
Tune will not run successive halving in this case.

Important: you should keep the order in which config is being created.
Example: config['dataset'] has to be set before config['tune'], since max_t of ASHA depends on the dataset.
"""

from ray import tune
import torch.nn as nn
from collections import OrderedDict

config = OrderedDict()

config['fixed_arch'] = True
config['replicate_chargrid'] = True # remove this in other config files or set to False
config['model_name'] = None  # placeholder, will get filled with sampled arch string

config['verbose'] = False
config['debug'] = False
config['sanity_checks'] = True # TODO: comment out after tests

###########
# DATASETS
###########
config['dataset'] = dict()

# placeholder, gets filled with dataset which is being specified via command line
config['dataset']['selected'] = None

# Cityscapes
config['dataset']['Cityscapes'] = dict()
config['dataset']['Cityscapes']['name'] = 'Cityscapes'
config['dataset']['Cityscapes']['n_classes'] = 20
config['dataset']['Cityscapes']['bg_classes'] = [0] # background n_cl do not contribute to key metric
config['dataset']['Cityscapes']['input_channels'] = 3
config['dataset']['Cityscapes']['height'] = 256
config['dataset']['Cityscapes']['width'] = 512
config['dataset']['Cityscapes']['max_epochs'] = 300
config['dataset']['Cityscapes']['samples_train'] = None  # None translates to use all
config['dataset']['Cityscapes']['samples_val'] = None  # None translates to use all
config['dataset']['Cityscapes']['samples_test'] = None  # None translates to use all
config['dataset']['Cityscapes']['batch_size_train'] = 7
config['dataset']['Cityscapes']['batch_size_val'] = 7

# Chargrid
config['dataset']['Chargrid'] = dict()
config['dataset']['Chargrid']['name'] = 'Chargrid'
config['dataset']['Chargrid']['n_classes'] = 92
config['dataset']['Chargrid']['bg_classes'] = [0] # background n_cl do not contribute to key metric
config['dataset']['Chargrid']['input_channels'] = 60
config['dataset']['Chargrid']['height'] = 336
config['dataset']['Chargrid']['width'] = 256
config['dataset']['Chargrid']['max_epochs'] = 100
config['dataset']['Chargrid']['samples_train'] = None  # None translates to use all
config['dataset']['Chargrid']['samples_val'] = None  # None translates to use all
config['dataset']['Chargrid']['samples_test'] = None  # None translates to use all
config['dataset']['Chargrid']['batch_size_train'] = 7
config['dataset']['Chargrid']['batch_size_val'] = 7


# uncomment the next two dictionary entries when training on a P100
# batch size of 16 is derived as max common batch size for both datasets on P100
# memory footprint Cityscapes: 16276MiB / 16280MiB 
# memory footprint Chargrid: ~15.1 GiB

#config['dataset']['Cityscapes']['batch_size_train'] = 16
#config['dataset']['Cityscapes']['batch_size_val'] = 16
#config['dataset']['Chargrid']['batch_size_train'] = 16
#config['dataset']['Chargrid']['batch_size_val'] = 16


########
# TUNE
########
config['tune'] = dict()

config['tune']['key_metric'] = 'mIoU_val'
config['tune']['num_samples'] = 1000
config['tune']['experiment_name'] = None
config['tune']['save_last'] = True


# HPs related to successive halving
config['tune']['ASHA'] = dict()
config['tune']['ASHA']['time_attr'] = 'training_iteration'
config['tune']['ASHA']['grace_period'] = 1
config['tune']['ASHA']['max_t'] = None
config['tune']['ASHA']['reduction_factor'] = 2
config['tune']['ASHA']['brackets'] = 1
config['tune']['ASHA']['metric'] = 'best_' + config['tune']['key_metric']
config['tune']['ASHA']['mode'] = 'max'

# model space
config['model'] = dict()

# non-arch related  HPs

# optimizer
config['model']['optimizer'] = dict()
config['model']['optimizer']['name'] = 'torch.optim.SGD'
config['model']['optimizer']['torch.optim.SGD_kwargs'] = dict()  # key will be called as name in eval(name(**kwargs))
config['model']['optimizer']['torch.optim.SGD_kwargs']['lr'] = 0.05
config['model']['optimizer']['torch.optim.SGD_kwargs']['momentum'] = 0.9
config['model']['optimizer']['torch.optim.SGD_kwargs']['weight_decay'] = 0.0001


"""
# also sample LR in future:
#config['model']['optimizer']['torch.optim.Adam_kwargs'] = dict()  # key will be called as name in eval(name(**kwargs))
config['model']['optimizer'] = dict()
config['model']['optimizer']['name'] = tune.choice(['torch.optim.SGD']) # change to e.g. .choice(['torch.optim.SGD','torch.optim.Adam'])
config['model']['optimizer']['torch.optim.SGD_kwargs'] = dict()  # key will be called as name in eval(name(**kwargs))
config['model']['optimizer']['torch.optim.SGD_kwargs']['lr'] = tune.choice([0.1])
config['model']['optimizer']['torch.optim.SGD_kwargs']['momentum'] = tune.choice([0.9])
config['model']['optimizer']['torch.optim.SGD_kwargs']['weight_decay'] = tune.choice([0.0001])
"""


# loss
# config['model']['loss_class_weighting_constant'] = tune.choice(list(range(1,11)) # turn later into choice([1,2,..10]) > sample and load precomputed class weights
config['model']['loss'] = nn.CrossEntropyLoss(weight=None)

# arch related  HPs
config['model']['arch'] = dict()
config['model']['arch']['dropout_ratio'] = 0.1
config['model']['arch']['momentum_bn'] = 0.1
config['model']['arch']['activation_function'] = nn.ReLU()
config['model']['arch']['padding_mode'] = 'reflect'
config['model']['arch']['supported_layers'] = 'CHVOTUD'
config['model']['arch']['n_base_channels'] = 64


# block 1
config['model']['arch']['b1_l1_op'] = 'D'
config['model']['arch']['b1_l2_op'] = 'C'
config['model']['arch']['b1_l3_op'] = 'C'
config['model']['arch']['b1_l1_dil'] = None
config['model']['arch']['b1_l2_dil'] = 1
config['model']['arch']['b1_l3_dil'] = 1

# block 2
config['model']['arch']['b2_l1_op'] = 'D'
config['model']['arch']['b2_l2_op'] = 'C'
config['model']['arch']['b2_l3_op'] = 'C'
config['model']['arch']['b2_l1_dil'] = None
config['model']['arch']['b2_l2_dil'] = 1
config['model']['arch']['b2_l3_dil'] = 1

# block 3
config['model']['arch']['b3_l1_op'] = 'D'
config['model']['arch']['b3_l2_op'] = 'C'
config['model']['arch']['b3_l3_op'] = 'C'
config['model']['arch']['b3_l1_dil'] = 1
config['model']['arch']['b3_l2_dil'] = 2
config['model']['arch']['b3_l3_dil'] = 2

# block 4
config['model']['arch']['b4_l1_op'] = 'C'
config['model']['arch']['b4_l2_op'] = 'C'
config['model']['arch']['b4_l3_op'] = 'C'
config['model']['arch']['b4_l1_dil'] = 4
config['model']['arch']['b4_l2_dil'] = 4
config['model']['arch']['b4_l3_dil'] = 4

# block 5
config['model']['arch']['b5_l1_op'] = 'C'
config['model']['arch']['b5_l2_op'] = 'C'
config['model']['arch']['b5_l3_op'] = 'C'
config['model']['arch']['b5_l1_dil'] = 8
config['model']['arch']['b5_l2_dil'] = 8
config['model']['arch']['b5_l3_dil'] = 8

# block 6
config['model']['arch']['b6_l1_op'] = 'O'
config['model']['arch']['b6_l2_op'] = 'T'
config['model']['arch']['b6_l3_op'] = 'C'
config['model']['arch']['b6_l4_op'] = 'C'
config['model']['arch']['b6_l1_dil'] = None
config['model']['arch']['b6_l2_dil'] = None
config['model']['arch']['b6_l3_dil'] = 1
config['model']['arch']['b6_l4_dil'] = 1

# block 7
config['model']['arch']['b7_l1_op'] = 'O'
config['model']['arch']['b7_l2_op'] = 'T'
config['model']['arch']['b7_l3_op'] = 'C'
config['model']['arch']['b7_l4_op'] = 'C'
config['model']['arch']['b7_l1_dil'] = None
config['model']['arch']['b7_l2_dil'] = None
config['model']['arch']['b7_l3_dil'] = 1
config['model']['arch']['b7_l4_dil'] = 1

# block 8
config['model']['arch']['b8_l1_op'] = 'O'
config['model']['arch']['b8_l2_op'] = 'T'
config['model']['arch']['b8_l3_op'] = 'C'
config['model']['arch']['b8_l4_op'] = 'C'
config['model']['arch']['b8_l1_dil'] = None
config['model']['arch']['b8_l2_dil'] = None
config['model']['arch']['b8_l3_dil'] = 1
config['model']['arch']['b8_l4_dil'] = 1




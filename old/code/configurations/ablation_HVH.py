from ray import tune
from configurations.chargrid import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

# uncomment the next two dictionary entries when training on a P100

#config['dataset']['Cityscapes']['batch_size_train'] = 16
#config['dataset']['Cityscapes']['batch_size_val'] = 16

#config['dataset']['Chargrid']['batch_size_train'] = 16
#config['dataset']['Chargrid']['batch_size_val'] = 16

config['fixed_arch'] = True
config['replicate_chargrid'] = False

# block 1
config['model']['arch']['b1_l2_op'] = 'H'
config['model']['arch']['b1_l3_op'] = 'V'


# block 2
config['model']['arch']['b2_l2_op'] = 'H'
config['model']['arch']['b2_l3_op'] = 'V'


# block 3
config['model']['arch']['b3_l2_op'] = 'H'
config['model']['arch']['b3_l3_op'] = 'V'


# block 4
config['model']['arch']['b4_l1_op'] = 'H'
config['model']['arch']['b4_l2_op'] = 'V'
config['model']['arch']['b4_l3_op'] = 'H'


# block 5
config['model']['arch']['b5_l1_op'] = 'H'
config['model']['arch']['b5_l2_op'] = 'V'
config['model']['arch']['b5_l3_op'] = 'H'


# block 6
config['model']['arch']['b6_l3_op'] = 'H'
config['model']['arch']['b6_l4_op'] = 'V'


# block 7
config['model']['arch']['b7_l3_op'] = 'H'
config['model']['arch']['b7_l4_op'] = 'V'


# block 8
config['model']['arch']['b8_l3_op'] = 'H'
config['model']['arch']['b8_l4_op'] = 'V'





from ray import tune
from configurations.chargrid import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

config['fixed_arch'] = False
config['replicate_chargrid'] = False

# as we sample a lot of models, we save disk space by only serializing the best found weights (per model)
config['tune']['save_last'] = False

# we search more aggressively
config['tune']['ASHA']['reduction_factor'] = 4

# and more/longer as we anticipate many architectures to fail because out of memory errors
config['tune']['num_samples'] = 10000

# only use 10k samples for Chargrid dataset
# the 10k samples will be selected by alphabetical order of file names in the train folder
config['dataset']['Chargrid']['samples_train'] = 10000

# sample batch size (not very elegant, but does its job)
config['dataset']['Chargrid']['batch_size_train'] = tune.choice(list(range(3,11)))
config['dataset']['Chargrid']['batch_size_val'] = deepcopy(config['dataset']['Cityscapes']['batch_size_train'])
config['dataset']['Cityscapes']['batch_size_val'] = deepcopy(config['dataset']['Cityscapes']['batch_size_train'])
config['dataset']['Cityscapes']['batch_size_val'] = deepcopy(config['dataset']['Cityscapes']['batch_size_train'])

# sample other HPs
config['model']['arch']['dropout_ratio'] = tune.uniform(0,0.5)
config['model']['arch']['momentum_bn'] = tune.uniform(0,0.5)
config['model']['arch']['n_base_channels'] = tune.choice(list(range(60,71))[::2]) # can only handle even number of base channels

# redefine search space 

# block 1
config['model']['arch']['b1_l1_op'] = 'D'
config['model']['arch']['b1_l2_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b1_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b1_l1_dil'] = None
config['model']['arch']['b1_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b1_l3_dil'] =  tune.choice(list(range(1,8)))

# block 2
config['model']['arch']['b2_l1_op'] = 'D'
config['model']['arch']['b2_l2_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b2_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b2_l1_dil'] = None
config['model']['arch']['b2_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b2_l3_dil'] = tune.choice(list(range(1,8)))

# block 3
config['model']['arch']['b3_l1_op'] = 'D'
config['model']['arch']['b3_l2_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b3_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b3_l1_dil'] = None
config['model']['arch']['b3_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b3_l3_dil'] = tune.choice(list(range(1,8)))

# block 4
config['model']['arch']['b4_l1_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b4_l2_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b4_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b4_l1_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b4_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b4_l3_dil'] = tune.choice(list(range(1,8)))

# block 5
config['model']['arch']['b5_l1_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b5_l2_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b5_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b5_l1_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b5_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b5_l3_dil'] = tune.choice(list(range(1,8)))

# block 6
config['model']['arch']['b6_l1_op'] = 'O'
config['model']['arch']['b6_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b6_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b6_l4_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b6_l1_dil'] = None
config['model']['arch']['b6_l2_dil'] = None
config['model']['arch']['b6_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b6_l4_dil'] = tune.choice(list(range(1,8)))

# block 7
config['model']['arch']['b7_l1_op'] = 'O'
config['model']['arch']['b7_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b7_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b7_l4_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b7_l1_dil'] = None
config['model']['arch']['b7_l2_dil'] = None
config['model']['arch']['b7_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b7_l4_dil'] = tune.choice(list(range(1,8)))

# block 8
config['model']['arch']['b8_l1_op'] = 'O'
config['model']['arch']['b8_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b8_l3_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b8_l4_op'] = tune.choice(['H','C','V'])
config['model']['arch']['b8_l1_dil'] = None
config['model']['arch']['b8_l2_dil'] = None
config['model']['arch']['b8_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b8_l4_dil'] = tune.choice(list(range(1,8)))

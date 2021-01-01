from ray import tune
from configurations.chargrid import config as base_config
from copy import deepcopy

config = deepcopy(base_config)


config['fixed_arch'] = False # if you set this to true, then you'll randomly sample 1 arch and train it for max_epochs
config['replicate_chargrid'] = False

# block 1
config['model']['arch']['b1_l1_op'] = 'D'
config['model']['arch']['b1_l2_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b1_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b1_l1_dil'] = None
config['model']['arch']['b1_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b1_l3_dil'] =  tune.choice(list(range(1,8)))

# block 2
config['model']['arch']['b2_l1_op'] = 'D'
config['model']['arch']['b2_l2_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b2_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b2_l1_dil'] = None
config['model']['arch']['b2_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b2_l3_dil'] = tune.choice(list(range(1,8)))

# block 3
config['model']['arch']['b3_l1_op'] = 'D'
config['model']['arch']['b3_l2_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b3_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b3_l1_dil'] = None
config['model']['arch']['b3_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b3_l3_dil'] = tune.choice(list(range(1,8)))

# block 4
config['model']['arch']['b4_l1_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b4_l2_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b4_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b4_l1_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b4_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b4_l3_dil'] = tune.choice(list(range(1,8)))

# block 5
config['model']['arch']['b5_l1_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b5_l2_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b5_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b5_l1_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b5_l2_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b5_l3_dil'] = tune.choice(list(range(1,8)))

# block 6
config['model']['arch']['b6_l1_op'] = 'O'
config['model']['arch']['b6_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b6_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b6_l4_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b6_l1_dil'] = None
config['model']['arch']['b6_l2_dil'] = None
config['model']['arch']['b6_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b6_l4_dil'] = tune.choice(list(range(1,8)))

# block 7
config['model']['arch']['b7_l1_op'] = 'O'
config['model']['arch']['b7_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b7_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b7_l4_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b7_l1_dil'] = None
config['model']['arch']['b7_l2_dil'] = None
config['model']['arch']['b7_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b7_l4_dil'] = tune.choice(list(range(1,8)))

# block 8
config['model']['arch']['b8_l1_op'] = 'O'
config['model']['arch']['b8_l2_op'] = tune.choice(['T','U'])
config['model']['arch']['b8_l3_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b8_l4_op'] = tune.choice(['O','H','C','V'])
config['model']['arch']['b8_l1_dil'] = None
config['model']['arch']['b8_l2_dil'] = None
config['model']['arch']['b8_l3_dil'] = tune.choice(list(range(1,8)))
config['model']['arch']['b8_l4_dil'] = tune.choice(list(range(1,8)))

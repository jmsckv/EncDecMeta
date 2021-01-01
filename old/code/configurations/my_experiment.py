from ray import tune
from configurations.chargrid_10k import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

"""
Define your own experiment by entries in config.
E.g.:
config['model']['arch']['b1_l1_op'] = tune.choice(['H','V'])
"""


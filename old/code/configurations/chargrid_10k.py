from ray import tune
from configurations.chargrid import config as base_config
from copy import deepcopy

config = deepcopy(base_config)

# we use 10k training samples instead of ~40k samples
# the 10k samples will be selected by alphabetical order of file names in the train folder
config['dataset']['Chargrid']['samples_train'] = 10000


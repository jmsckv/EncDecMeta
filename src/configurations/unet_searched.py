"""
A unet search space in which unet.py is embedded and at the upper range of possible parameters, i.e. we search for more lightweigt bute equally performant models compared to the baseline.
We search both, hyperparameters as well as architectures.
Note that the batch size might be higher than in the baseline model (batch size = 1).

"""

c = (['H','V','V','O'], range(1,8))

config = {'experiment_name': 'unet_searched',
'D_blocks': [[c],[c],[c],[c],[]],
'B_blocks': [[c]], # could als be formulated as [[c],[c]]
'U_blocks': [[c],[c],[c],[c], [c,c]],
'H': 256,  # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': (0,0.5),
'momentum': (0.5,1),
'momentum_bn': (0,1),
'lr': [i*j for i in [1,3,5,7] for j in [0.1, 0.01, 0.001]],
'weight_decay': [i*j for i in [1,3,5,7] for j in [0.01, 0.001, 0.0001]],
'nesterov': [True,False],
'base_channels': range(32,65), 
'batch_size': range(1,20)}
             









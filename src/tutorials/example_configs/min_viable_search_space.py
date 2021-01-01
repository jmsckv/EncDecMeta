"""
A minimum viable search space to test.
Should work on a laptop w/o GPU.
"""

# formulate repetitive components outside of dictionary
choice = (['H','C','V','O'], range(1,10))

config = {'experiment_name': 'min_search',
'D_blocks': [[]],
'U_blocks': [[choice]],
'B_blocks': [[choice,choice],[choice]],
'H': 256, # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
'momentum': (0, 0.5),
'momentum_bn': (0, 1),
'lr': [i*j for i in [1,3,5] for j in [0.01, 0.1, 0.001]],
'weight_decay': (0.0001, 0.01),
'nesterov': [True,False],
'base_channels': 5, 
'batch_size': 3,
'test_run': 6,
'cpus_per_trial': 1,
'max_epochs': 10,
'num_samples': 10,
'checkpoint_freq':1} # remove or change to None to keep for each model only best weights



"""
A minimum viable config file to train a fixed architecture.
We reduce training and test set to 4 samples each via <test_run:6>.

What you may want to additionally try out:

- <'checkpoint_freq':1> which will serialize every model after every epoch

- <num_samples:X> with X>1 to observe that now a search will be triggered. Of course, this only makes sense if you specify a search space in the above dictionary. 
Else the same architecture will be trained X times. This still may be a desirable behavior if you want to repeat the same experiment with different random seeds.

- modify 'H'/'W' to a e.g. half their previous values. This will interpolate the resolution of the data/labels to the specified resolution.
If you specify 'H', 'W' equal to the actual image size, no interpolation will take place. 

- <'overfit:True> which will reduced the training and validation set to the same single sample; given sufficent model capacity, all metrics should converge to the best possible scenario
This is to check that gradient updates work properly.

Note:
- if a GPU is available, each trial will automatically be allocated on GPU, if <'cpus_per_trial'> is omitted, each trial will be allocated 4 CPUs
- to further find out about default values that can be altered in the config file run from a terminal inside the default Docker: cd $CODEPATH/src/; grep -r config.get

"""

config = {'experiment_name': 'min_viable_fixed_arch', 
'D_blocks':[[]],
'U_blocks':[[]],
'B_blocks': [[('C',3)]],
'H': 1024,
'W': 2048,
'dropout_ratio': 0.1,
'momentum': 0.2,
'momentum_bn': 0.1,
'lr': 0.01,
'weight_decay': 0.001,
'nesterov': True,
'base_channels': 8,
'batch_size' : 2,
'test_run': 4,
'cpus_per_trial': 1, 
'max_epochs':20,
'verbose':True}


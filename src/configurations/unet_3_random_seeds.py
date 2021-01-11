"""
Same as unet.py - but: this time we show how to run the same experiment with 3 random seeds.

"""
c = ('C', 3)
config = {'experiment_name': 'unet_3seeds_bs5',
'D_blocks': [[c],[c],[c],[c],[]],
'B_blocks': [[c]], # could als be formulated as [[c],[c]]
'U_blocks': [[c],[c],[c],[c], [c,c]],
'H': 256,  # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': 0.1,
'momentum': 0.99,
'momentum_bn': 0.1,
'lr': 0.01,
'weight_decay': 0.001,
'nesterov': False,
'base_channels': 64, 
'batch_size': 5,
'checkpoint_freq:': 1,
'max_t' : 500,
'num_samples' : 3}
             









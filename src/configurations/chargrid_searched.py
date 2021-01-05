"""
An encoder-decoder search space in which the proposed in "Chargrid: Towards Understanding 2D documents" by Katti et al. (2018) - https://arxiv.org/abs/1809.08799 - is embedded.
Here, we only include the decoder head responsible for semantic segmentation.
Another difference is that we do not us any data augmentation.
Further, also non-architectural hyperparameters are being searched. 
"""

c = (['H','V','C','O'], range(1,8))
b = [c,c]
bb = [c,c,c]

config = {'experiment_name': 'chargrid_searched',
'D_blocks': [b,b,b],
'B_blocks': [bb,bb],
'U_blocks': [b,b,b],
'H': 256,  # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': (0,0.5),
'momentum': (0.5,1),
'momentum_bn': (0,1),
'lr': [i*j for i in [1,3,5,7] for j in [0.1, 0.01, 0.001]],
'weight_decay': [i*j for i in [1,3,5,7] for j in [0.01, 0.001, 0.0001]],
'nesterov': [True,False],
'base_channels': [46,64,80], 
'batch_size': range(3,20),
'max_epochs': 300,
'num_samples': 1000} 









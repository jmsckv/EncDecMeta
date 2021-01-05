"""
An encoder-decoder model as proposed in "Chargrid: Towards Understanding 2D documents" by Katti et al. (2018) - https://arxiv.org/abs/1809.08799.
Here, we only include the decoder head responsible for semantic segmentation.

Below, with <max_epochs:10000> and <overfit:True>, we overfit on a single training sample.
We want to see all metrics converge.

"""

c1 = ('C', 1)
c2 = ('C', 2)
c4 = ('C', 4)
c8 = ('C', 8)

config = {'experiment_name': 'chargrid_fixed',
'D_blocks': [[c1,c1],[c1,c1],[c2,c2]],
'B_blocks': [[c4,c4,c4],[c8,c8,c8]],
'U_blocks': [[c1,c1],[c1,c1],[c1,c1]],
'H': 256,  # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': 0.1,
'momentum': 0.9,
'momentum_bn': 0.1,
'lr': 0.05,
'weight_decay': 0.001,
'nesterov': False,
'base_channels': 64, 
'batch_size': 7} # differs from orig paper 









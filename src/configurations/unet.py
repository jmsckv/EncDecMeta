"""

An encoder-decoder model as proposed in "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. (2015) - https://arxiv.org/abs/1505.04597.

_________________________________________

Differences Architecture:


Note that other than in the original paper, we do not make any use of bilinear/bicubic downsampling:
- neither to reduce spatial resolution > we use transposed convloutions for that
- nor to adjust the size of feature maps being fused via skip connections > this frameworks guarantees that the incoming feature maps are of the same resolution

Since we have learnable parameters in the transposed convolution, we only use one additional layer to model a convloutional block.

Unet Paper Down Conv. Block: Downsampling 2x2 Max Pooling > 3x3 Conv. > 3x3 Conv
Here: 3x3 Conv with Stride 2 > 3x3 Conv


Upsampling in the original paper takes place as follows:
" At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU."

Here we directly apply a 3*3 Transposed Convolution (relatively more parameters) followed by a 1x1 depthwise convolution (relatively fewer parameters).

Unet Paper Up Conv. Block: Upsampling >  2x2 Conv > 3x3 Conv. (> 3x3 in last block)
Here: 3x3  Transposed Conv with Stride 2 > 1x1 Conv > 3x3 Conv.  (> 3x3 in last block)

_________________________________________


Ambiguities HPs:


Note that from the original paper it is not clear how dropout is concretley specified:
" Drop-out layers at the end of the contracting path perform further implicit
data augmentation."

Here, we set the dropout to 0.1 throughout the network, the momentum parameter of the BN is set as the momentum of the optimizer.(For the BN momentum is inverseley formulated in Pytorch: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

Also note that in the original paper also the learning rate is not being reported.
Here we set it 0.01. We set the unreported weight decay to: 0.001

"""

c = ('C', 3)

config = {'experiment_name': 'unet_fixed',
'D_blocks': [[c],[c],[c],[c],[]],
'B_blocks': [[c]], # could als be formulated as [[c],[c]]
'U_blocks': [[c],[c],[c],[c], [c,c]],
'H': 256,  # downsampling orig Cityscapes by factor 4
'W': 512,  # downsampling orig Cityscapes by factor 4
'dropout_ratio': 0,
'momentum': 0.99,
'momentum_bn': 0.1,
'lr': 0.01,
'weight_decay': 0.001,
'nesterov': False,
'base_channels': 64, 
'batch_size': 1}
             









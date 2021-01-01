from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import random

############
# CONV LAYER 
############

def get_conv_layer(conv_type, conv_spec:dict, config:dict, out_conv=False):
    """
    Get a standard convolutional layer, where convolutions are followed by batch norm and activation layer.
    This function can also be applied to a transposed and one-by-one (depthwise) convolution.
    Do not apply this a conv layer that replaces a fully connected layer (e.g. last layer in network returning logits.)
    """

    if out_conv:
        return nn.Sequential(conv_type(**conv_spec))
    else:
        return nn.Sequential(
            conv_type(**conv_spec),
            nn.BatchNorm2d(num_features=conv_spec['out_channels'], momentum=config['momentum_bn']),
            config['activation_function'],
            nn.Dropout(config['dropout_ratio']))


########
# LAYER
########

def get_layer(operation:str, in_channels:int, out_channels:int, config:dict, dilation:int=1, out_conv:bool=False):
    """ Get torch.nn representation of a layer presentation."""

    if config['debug'] >=1 : assert operation in 'DUOHCV', 'Unsupported operation type.'
        
    conv_spec = {}
    conv_spec['in_channels'] = in_channels
    conv_spec['out_channels'] = out_channels

    if operation == 'U':  # layer type 1: "U"psampling via 3*3 transpose convolution with stride 2

        operation_translated =  nn.ConvTranspose2d
        conv_spec['padding_mode'] = 'zeros'
        conv_spec = {**conv_spec,**{'dilation': 1, 'stride': 2, 'kernel_size': 3, 'padding': 1, 'output_padding': 1}}

    else:
        operation_translated = nn.Conv2d
        conv_spec['padding_mode'] = config['padding_mode']

        if operation == 'D': #layer type 2: "D"ownsampling via  3*3 convolution with stride 2
            conv_spec = {**conv_spec,**{'stride': 2, 'kernel_size': 3, 'padding': 1, 'dilation': 1}}

        # layer type 3: {1*1,3*3,1*3,3*1} convolutional layers that keep spatial resolution constant


        if operation == 'O': # O: One-by-one (depthwise) convolution
            conv_spec = {**conv_spec,**{'kernel_size': 1, 'dilation': 1}}

        if operation == 'C': # C: 3*3 convolution
            conv_spec = {**conv_spec,**{'dilation': dilation, 'kernel_size': 3, 'padding': dilation}}

        if operation == 'H':  # H: 1*3, 'horizontal' convolution
            conv_spec = {**conv_spec,**{'dilation': (1, dilation), 'kernel_size': (1, 3), 'padding': (0, dilation)}}

        if operation == 'V': # V: 3*1, 'vertical' convolution
            conv_spec = {**conv_spec,**{'dilation': (dilation, 1), 'kernel_size': (3, 1), 'padding': (dilation, 0)}}
            
    return get_conv_layer(operation_translated,conv_spec,config)



################################
# SAMPLE BLOCK, RETURN AS NN
################################


def get_block(block_type:str, block_number:int, in_channels:int, out_channels:int,  sampled_layers:List[Tuple[str,int]], config:dict):
    """Get torch.nn represesentation of either downsampling, upsampling, or bottleneck block."""

    if config['debug'] >= 1:
        assert block_type in "DBU", 'Unknown block type, must be one of "D"(ownsampling), "B"(ottleneck), "U"(psampling).'
        if block_type == "B": 
            assert len(sampled_layers) >= 1 , "In bottleneck blocks at least one layer must get sampled."
    
    layers = nn.ModuleList()

    arch_string = block_type + str(block_number) + '('

    if block_type == "D":
        if block_number == 0:
            in_channels = config['channels']
            out_channels = config['base_channels']
        layers += get_layer('D',in_channels,out_channels,config)
    
    elif block_type == "U":
        layers += get_layer('O',in_channels,out_channels,config)
        in_channels = out_channels
        layers += get_layer('U',in_channels,out_channels,config)

    elif block_type == 'B': 
        if config['debug'] >= 1:
            assert len(sampled_layers[0]) == 2 , 'For bottleneck blocks you have to specify at least one layer.'
        operation,dilation = sampled_layers.pop(0)
        layers += [get_layer(operation,in_channels,out_channels,config,dilation)]
        arch_string +=  str(operation) + str(dilation)
    
    in_channels = out_channels

    if len(sampled_layers):
        for l in sampled_layers:
            if config['debug'] >= 1: 
                assert isinstance(l,tuple), 'Layers must be encoded as tuples.'
                assert len(l) == 2, 'Layers must be encoded as tuples with two entries'
                assert isinstance(l[0],str), 'First tuple entry must be a string encoding layer operation.'
                assert isinstance(l[1],int), 'Second tuple entry must be an int encoding  dilation rate.'

            operation, dilation = l[0],l[1]
            arch_string +=  str(operation) + str(dilation)
            layers += [get_layer(operation,in_channels,out_channels,config,dilation)]

    arch_string += ')'

    if config['debug'] >= 1:
        assert isinstance(layers[-1], nn.Module)
    if config['debug'] > 1:
        print(f'Sampled this block {arch_string}')

    return nn.Sequential(*layers), arch_string



"""
# Demo Setup 
cfg = {}
cfg['debug'] = 2 # 0 = deactivated, 1 = assertions-checks , 2 = assertion-checks & print-outs
cfg['dropout_ratio'] = [0.1, 0.9]
cfg['momentum_bn'] = [0.2, 0.8]
cfg['activation_function'] = [nn.ReLU(), nn.LeakyReLU(negative_slope=0.05)] # note: if you sample only LeakyReLU() you will use its default hyperparameters
cfg['padding_mode'] = ['zeros','replicate']  # zeros', 'reflect', 'replicate', 'circular'
cfg['supported_layers'] = 'CHBVOTUD'
cfg['base_channels'] = [1,32]
cfg['channels'] = [1,3,50]
cfg['classes'] = [1,20]
cfg['DownBlocks'] = [['HVO',[1,2]],['HVO',[1,2]]]
cfg['UpBlocks'] = [['VH',[1,2111]],]
cfg['BottleneckBlocks'] = [['CV',[1,32]],]
cfg['in_channels'] = [1,2]
cfg['out_channels'] =  [1,2,120]
cfg['dilation'] =  [1,3]
sampled_config = sample_from_config(cfg)
"""

""" 
# Demo: instantiate layer from config
for i in 'DUCVOH':
    l = get_layer(i,sampled)
    print(l)
"""

""" 
# Demo: instantiate a block
# commented out: for down- & upsampling blocks no additional layers must be sampled
for block_type in 'DU': # 'DU'
    for block_number in [0,4]:
            sampled_layers = [(random.choice('OHCV'),random.randint(0,10))for i in range(random.randint(2,7))]
            #sampled_layers = []
            block, arch = sample_block(block_type,block_number,sampled_layers,sampled_config)
            print(block,arch)
"""

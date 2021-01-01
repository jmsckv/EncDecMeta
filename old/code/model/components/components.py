from typing import Dict, List, Optional
import torch.nn as nn
from utils.sampling import sample_flat_hp

############
# CONV LAYER
############

def get_conv_layer(config, conv_type, in_channels, out_channels, out_conv=False, **kwargs):
    """
    Get a standard convolutional layer, where convolutions are followed by batch norm and activation layer.
    This function can also be applied to a transposed and one-by-one (depthwise) convolution.
    Do not apply this a conv layer that replaces a fully connected layer (e.g. last layer in network returning logits.)
    """
    padding_m = sample_flat_hp(config,'padding_mode', arch_hp=True)
    if conv_type is nn.ConvTranspose2d:
        padding_m = 'zeros'
    if not out_conv:
        return nn.Sequential(
            conv_type(in_channels, out_channels, padding_mode=padding_m, **kwargs),
            nn.BatchNorm2d(num_features=out_channels, momentum= sample_flat_hp(config, 'momentum_bn', arch_hp=True)),
            sample_flat_hp(config, 'activation_function', arch_hp=True),
            nn.Dropout(sample_flat_hp(config, 'dropout_ratio', arch_hp=True)))
    else:
        return nn.Sequential(conv_type(in_channels, out_channels, padding_mode=padding_m, **kwargs))

########
# LAYER
########

def get_layer(operation:str, in_channels:int, out_channels:int, config:dict, dilation=1, out_conv=False):
    currently_covered = 'CHVOTUD' # layer operations currently covered by this function
    assert len(set(currently_covered).difference(set(config['model']['arch']['supported_layers']))) is 0
    assert operation in currently_covered, 'Unsupported operation type.'
    in_channels = int(in_channels) # required as interaction with sampling in Tune leads to floats
    out_channels = int(out_channels)

    # layer type 1: upsampling layers {bilinear upsampling, 3*3 transpose convolution with stride 2}
    # U: bilinear upsamling
    if operation == 'U':
        return nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False)
    # T: transpose convolution
    elif operation == 'T':
        return get_conv_layer(config, nn.ConvTranspose2d, in_channels, out_channels,
                              **{'dilation': 1, 'stride': 2, 'kernel_size': 3, 'padding': 1, 'output_padding': 1})

    # layer type 2: downsampling layer
    # currently, and as in original Chargrid paper, downsampling is only supported via a 3*3 conv with stride 2
    # D: downsampling
    elif operation == 'D':
        return get_conv_layer(config, nn.Conv2d, in_channels, out_channels,
                              **{'stride': 2, 'kernel_size': 3, 'padding': 1, 'dilation': 1})

    # layer type 3: {1*1,3*3,1*3,3*1} conv layers that keep spatial resolution
    # dilation rate can vary in range 1-8
    # O: One-by-one (depthwise) convolution
    elif operation == 'O':
        return get_conv_layer(config, nn.Conv2d, in_channels, out_channels, **{'kernel_size': 1, 'dilation': 1})
    # C: 3*3 convolution
    elif operation == 'C':
        return get_conv_layer(config, nn.Conv2d, in_channels, out_channels, out_conv = out_conv,
                              **{'dilation': dilation, 'kernel_size': 3, 'padding': dilation})
    # H: 1*3, 'horizontal' convolution
    elif operation == 'H':
        return get_conv_layer(config, nn.Conv2d, in_channels, out_channels,
                              **{'dilation': (1, dilation), 'kernel_size': (1, 3), 'padding': (0, dilation)})
    # V: 3*1, 'vertical' convolution
    elif operation == 'V':
        return get_conv_layer(config, nn.Conv2d, in_channels, out_channels,
                              **{'dilation': (dilation, 1), 'kernel_size': (3, 1), 'padding': (dilation, 0)})
    else:
        raise NotImplementedError


################################
# SAMPLE BLOCK, RETURN AS NN
################################

def sample_block_and_return_nn(in_channels: int, out_channels: int, sampled_ops: List[str], sampled_dils: List[Optional[int]],config: Dict):
    assert len(sampled_ops) == len(sampled_dils)
    arch_string = ''
    layers = nn.ModuleList()
    for i in range(len(sampled_ops)):
        if i > 0: in_channels = out_channels  # upsampling/downsampling always in second layer
        layers.append(get_layer(operation=sampled_ops[i],
                                in_channels=in_channels,
                                out_channels=out_channels,
                                dilation=sampled_dils[i],
                                config = config))
        arch_string += sampled_ops[i]
        if  sampled_ops[i] in 'CDH':
            if sampled_dils[i] and sampled_dils[i] > 1 :
                arch_string += str(sampled_dils[i])
        arch_string += '|'
    arch_string = arch_string[:-1]
    assert isinstance(layers[-1], nn.Module)
    return nn.Sequential(*layers), arch_string





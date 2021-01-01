import torch
import torch.nn as nn

def sample_flat_hp(config, hp, arch_hp= False):
    if arch_hp:
        return config['model']['arch'][hp]
    else:
        return config['model'][hp]

def sample_hp_with_kwargs(config, hp, arch_hp=False, external_kwargs={}):
    if arch_hp:
        assert isinstance(config['model']['arch'][hp],dict)
        sampled = config['model']['arch'][hp]['name']
    else:
        assert isinstance(config['model'][hp],dict)
        sampled = config['model'][hp]['name']
    assert isinstance(sampled,str)
    try:
        if arch_hp:
            internal_kwargs = config['model']['arch'][hp][sampled + '_kwargs']
        else:
            internal_kwargs = config['model'][hp][sampled + '_kwargs']
    except:
        internal_kwargs = {}
    assert len(set(internal_kwargs.keys()).intersection(set(external_kwargs.keys()))) is 0
    kwargs = {**external_kwargs, **internal_kwargs}
    assert len(kwargs) > 0, 'No key word arguments contained in dict. Maybe simply eval(hp) instead of using this function?'
    return eval(sampled)(**kwargs)

# TODO: sample nested optimizer


"""
# tests:
# loss:
import torch
import torch.nn as nn
from configurations.chargrid import config
weights = torch.tensor([1,2,3])
hp = sample_flat_hp(config, 'loss')
print(hp), print(type(hp))
# arch
for i in ['batch_size_train','momentum_bn','activation_function']:
    hp = sample_flat_hp(config, i, arch_hp=True)
    print(hp), print(type(hp))
# optimizer
n = torch.nn.Conv2d(1,2,3)
hp = sample_hp_with_kwargs(config, 'optimizer', external_kwargs={'params':nn.Sequential(nn.Conv2d(1,1,2),nn.Conv2d(1,1,2)).parameters()})
print(hp), print(type(hp))
# block 1
hp = sample_flat_hp(config, 'b1_l1_op',is_arch_hp=True)
print(hp), print(type(hp))
hp = sample_flat_hp(config, 'b1_l2_op',is_arch_hp=True)
print(hp), print(type(hp))
hp = sample_flat_hp(config, 'b1_l1_dil',is_arch_hp=True)
print(hp), print(type(hp))
hp = sample_flat_hp(config, 'b1_l2_dil',is_arch_hp=True)
print(hp), print(type(hp))
"""
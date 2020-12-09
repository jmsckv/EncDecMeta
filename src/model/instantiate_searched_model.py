import torch.nn as nn
import torch
from model.components.components import get_layer, get_block

class EncDec(nn.Module):
    """
    Encodes an encoder decoder meta search space abstracted by blocks consisting of layers.
    Each architecture has to consists of at least one downsamling, one encoder, and one upsampling block.
    Each architecture has to consists of an equal number of downsampling and upsampling blocks.
    The architecture is encoded in config['sampled_blocks'][{'D','B','U'}] as a nested list of tuples.
    E.g. config['sampled_blocks']['D']= [[('H',2),('C,1')],[('V',1)],[('V',2)]] would encode 3 downsampling blocks, the first one consisting of 2 layers, the second and third one of 1 layer.
    """

    def __init__(self,  config: dict):
        super().__init__()
        self.config = config
        self.arch_string = []

        self.nn_down = nn.ModuleList()
        self.nn_bottle = nn.ModuleList()
        self.nn_up = nn.ModuleList()
        self.nn_logits = nn.ModuleList()

        if self.config['debug'] >= 1:
            for i in ['D_blocks', 'U_blocks', 'B_blocks']:
                assert len(self.config[i]) >= 1, "Only networks are allowed with at least one downsamling, bottleneck, and decoder block each." # TODO: drop requirement for bottleneck block
            assert len(self.config['D_blocks']) == len(self.config['U_blocks']), "You must specify an equal number of encoder and decoder blocks."

        # TODO?: test that given spatial input resulution checks (via formula?) if downsampling is possible as implied by number of downsampling blocks


        exp_f = 2  # expansion factor > double/halve channels in each encoder/decoder block

        # Downsampling blocks
        for i,b in enumerate(self.config['D_blocks']):
            channels_in = int(self.config['base_channels'] *  exp_f ** (i-1) )  # will get ignored in first downsampling block
            channels_out = int(self.config['base_channels'] * exp_f ** i) # will get ignored in second downsampling block
            b,a = get_block('D', i, channels_in, channels_out, b, self.config)
            self.nn_down += [b]
            self.arch_string+= [a]
            if  self.config['debug'] > 1:
                print(f'Appended this downsampling block to overall network topology: {a,b}:')

    # Bottlenecks blocks
        for i,b in enumerate(self.config['B_blocks']):
            channels_in = channels_out # spatial resolution and number of channels remains unchanged
            b,a = get_block('B', i, channels_in, channels_out, b, self.config)
            self.nn_bottle += [b]
            self.arch_string += [a]
            if  self.config['debug'] > 1:
                print(f'Appended this bottleneck block to overall network topology: {a,b}:')

        # Upsampling blocks
        for i,b in enumerate(self.config['U_blocks']):
            channels_in = int(channels_out*2) # spatial resolution and number of channels remains unchanged
            channels_out = int(channels_in/4)
            if i == len(self.config['U_blocks']) -1 :
                channels_out = self.config['base_channels']
            b,a = get_block('U', i, channels_in, channels_out, b, self.config)
            self.nn_up += [b]
            self.arch_string += [a]
            if  self.config['debug'] > 1:
                print(f'Appended this upsampling to overall network topology: {a,b}:')

        # here we could place an upsampling operation if we take an downsampled input but want to predict full resolution

        # last block is currently fixed: out-conv, returning logits
        self.nn_logits += [get_layer('C', self.config['base_channels'], self.config['classes'], self.config, out_conv=True)]

        # turn arch string list into arch string: blocks are seperated by '*'
        self.arch_string = '*'.join(self.arch_string)
        self.config['arch_string'] = self.arch_string 
    
        if  self.config['debug'] > 1:
            print(f'Sampled this architecture: {self.arch_string}')


    def forward(self, x):

        if self.config['debug'] >= 1:
            for i in [2,3]:
                assert x.shape[i] % 2 == 0, 'Input resolution must be even before every downsampling block.'

        tmp_res = [] # store intermediary results of encoder

        # encoder
        for i,b in enumerate(self.nn_down):
            if self.config['debug'] > 1: print(f'Downsampling block number {i}, ingoing tensor shape {x.shape}')
            x = b(x)
            tmp_res += [x]
            if self.config['debug'] > 1: print(f'Downsampling block number {i}, outgoing tensor shape {x.shape}')
            
            if self.config['debug'] >= 1:
                if i < len(self.nn_down)-1: 
                    for j in [2,3]:
                        assert x.shape[j] % 2 == 0, 'Input resolution must be even before every downsampling block.'


        # bottleneck
        for i,b in enumerate(self.nn_bottle):
            if self.config['debug'] > 1: print(f'Bottleneck block number {i}, ingoing tensor shape {x.shape}')
            x = b(x)
            if self.config['debug'] > 1: print(f'Bottleneck block number {i}, outgoing tensor shape {x.shape}')
            
   
        # decoder
        for i,b in enumerate(self.nn_up):
            if self.config['debug'] > 1: print(f'Upsampling block number {i}, ingoing tensor shape {torch.cat([x, tmp_res[::-1][i]], dim=1).shape}')
            x = b(torch.cat([x, tmp_res[::-1][i]], dim=1))
            if self.config['debug'] > 1: print(f'Upsampling block number {i}, outgoing tensor shape {x.shape}')


        logits = self.nn_logits[0](x)
        if self.config['debug'] > 1 : print(logits.shape)

        return logits


        






















"""


# layer = (operation,dilation)
# block = [layer,layer,layer]
# D/B/U = [block,block,block]



# Demo 1 
cfg = {}
cfg['model'] = {}
cfg['dropout_ratio'] = 0.1
cfg['momentum_bn'] = 0.1
cfg['activation_function'] = nn.ReLU() # nn.
cfg['padding_mode'] = 'zeros' # zeros
cfg['base_channels'] = 8
cfg['channels'] = 3
cfg['classes'] = 7
cfg['debug'] = 2
complicated_block = [('H',3),('V',3),('C',4)] * 2
cfg['D_blocks'] = [[],[],complicated_block] * 3
cfg['B_blocks'] = [[('C',6)],[('H',3)]]
cfg['U_blocks'] = [[('H',2)],[],[('V',4)]] * 3
e = EncDec(cfg)
t = torch.ones(size=[1, 3, 256*4, 512*4]) # *4 * 4
o = e(t)
print(o.shape)

# Demo 2 - Chargrid arch
# deviating from base channel allocation


c1 = ('C', 1)
c2 = ('C', 2)
c4 = ('C', 4)
c8 = ('C', 8)
cfg['D_blocks'] = [[c1,c1],[c1,c1],[c2,c2]] 
cfg['B_blocks'] = [[c4,c4,c4],[c8,c8,c8]]
cfg['U_blocks'] =  [[c1,c1],[c1,c1],[c1,c1]] 
e = EncDec(cfg)
t = torch.ones(size=[1, 3, 256*4, 512*4]) # *4 * 4
o = e(t)
print(o.shape)


"""


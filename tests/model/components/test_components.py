import unittest
import torch
import torch.nn as nn
import random

from utils.test_generator import EncDecTestGenerator
from model.components.components import get_layer, get_block

class TestGetLayer(unittest.TestCase):

    def setUp(self):
        self.tg = EncDecTestGenerator(quicktest=True)
        self.tg.cfg_var['in_channels']=[1,7,64]
        self.tg.cfg_var['out_channels']=[1,2,64]
        self.tg.cfg_var['H']=[2,512]
        self.tg.cfg_var['W']=[2,64]
        self.tg.cfg_var['dilation']=[1,7,64]


    def test_get_layer(self):
        # Testing relation of shapes of input and output tensors.
        cnt = 0
        for i in self.tg.cfg_fixed['operation_types']:
            for j in self.tg.generate_tests:
                with self.subTest('Testing layer operation:', i=i, j=j):
                    net = get_layer(i, j['in_channels'],j['out_channels'],j,j['dilation'])
                    t = EncDecTestGenerator.get_ingoing_tensor(j)
                    out = net(t)
                    self.assertEqual(out.size()[1],j['out_channels'])
                    if i == 'U':
                        self.assertEqual(out.size()[2],j['H']*2)
                        self.assertEqual(out.size()[3],j['W']*2)
                    elif i == 'D':
                        self.assertEqual(out.size()[2]*2,j['H'])
                        self.assertEqual(out.size()[3]*2,j['W'])
                    elif i == 'B':
                        self.assertEqual(t.size()[2],j['H'])
                        self.assertEqual(t.size()[3],j['W'])
                    cnt += 1
        print(f'Tested {cnt} combinations.') # 648



class TestGetBlock(unittest.TestCase):
    """
    Test behavior of blocks: 
    - is a valid torch.nn derived from (sampled) entries in the config,
    - are wrong specifications in the config refuse,
    - are shapes of inputs tensors mapped to expected output tensor shapes?
    """
    
    def setUp(self):
        self.tg = EncDecTestGenerator(quicktest=False)
        self.tg.cfg_var['block_number'] = [0,5]
        self.tg.cfg_var['in_channels']=[1,7]
        self.tg.cfg_var['out_channels']=[1,64]
        self.tg.cfg_var['H']=[2,512]
        self.tg.cfg_var['W']=[2,64]
        self.tg.cfg_var['dilation']=[1,7,64]


    def test_get_block_B_misspecified(self, block_type='B'):
        """
        Layers must follow a predefined layout in the config: List[Tuple(str,int)].
        """
        cnt = 0
        for error in [[],[()],[('V',2,3)],[('V',1),('H',1,2)]]:
            for i in self.tg.generate_tests:
                with self.subTest('Testing:', block_type=block_type,i=i,error=error):
                    with self.assertRaises(AssertionError): #change to ValueError to see that test is working
                        cnt += 1
                        net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],error,i)
        print(f'Tested {cnt} combinations.') # 384
        

    def test_get_block_B(self, block_type='B'):
        """
        Test correct behavior for B blocks: H & W remains unchanged.
        """
        self.tg.cfg_var['H'] += [3] # other than D & U, B block can also handle uneven W/H that are not multiples of 2 
        self.tg.cfg_var['W'] += [27] 
        cnt = 0
        for i in self.tg.generate_tests:
            sampled_layers = [(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))]
            with self.subTest('Testing:', block_type=block_type,i=i,sampled_layers=sampled_layers):
                in_tensor = EncDecTestGenerator.get_ingoing_tensor(i)
                net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],sampled_layers,i)
                out_tensor = net(in_tensor)
                self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]) # same H
                self.assertEqual(in_tensor.size()[3],out_tensor.size()[3]) # same W 
                cnt += 1
        print(f'Tested {cnt} combinations.') # 216


    def test_get_block_D0(self, block_type='D'):
        """
        Test correct behavior for first D block: channels_in, channels_out are overwritten with config['classes'], config['base_channels']
        """
        self.tg.cfg_var['block_number'] = [0]
        cnt = 0
        for i in self.tg.generate_tests:
            for j in [True,False]: # D & U blocks should work without additionally sampling layers
                if j: 
                    sampled_layers = [(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))]
                else: 
                    sampled_layer = []
                with self.subTest('Testing:', block_type=block_type,i=i,sampled_layers=sampled_layers):
                    in_tensor = EncDecTestGenerator.get_input_tensor(i)
                    net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],sampled_layers,i)
                    out_tensor = net(in_tensor)
                    self.assertEqual(in_tensor.size()[1],i['channels'])
                    self.assertEqual(out_tensor.size()[1],i['base_channels'])  
                    self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]*2) 
                    self.assertEqual(in_tensor.size()[3],out_tensor.size()[3]*2) 
                    cnt += 1
        print(f'Tested {cnt} combinations.') # 96


    def test_get_block_D(self, block_type='D'):        
        """
        Test correct behaviour for D block > 1: resolution gets halved, channels doubled. Should work for all even input resolutions. 
        """
        self.tg.cfg_var['block_number'] = [1,7]
        self.tg.cfg_var['H'] += [28,300] # other than D & U, B block can also handle uneven W/H that are not multiples of 2 
        self.tg.cfg_var['W'] += [14] 
        cnt = 0
        for i in self.tg.generate_tests:
            for j in [True,False]: # D & U blocks should work without additionally sampling layers
                if j: 
                    sampled_layers = [(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))]
                else: 
                    sampled_layer = []
                with self.subTest('Testing:', block_type=block_type,i=i,sampled_layers=sampled_layers):
                    in_tensor = EncDecTestGenerator.get_ingoing_tensor(i)
                    i['out_channels'] = i['in_channels']*2
                    net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],sampled_layers,i)
                    out_tensor = net(in_tensor)
                    self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]*2) 
                    self.assertEqual(in_tensor.size()[3],out_tensor.size()[3]*2)
                    self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]*2) 
                    self.assertEqual(in_tensor.size()[1]*2,out_tensor.size()[1]) 
                    cnt += 1
        print(f'Tested {cnt} combinations.') 


    def test_get_block_U(self, block_type='U'):        
        """
        Test correct behaviour for U blocks: resolution gets doubled, channels halved. Should work for all input resolutions, and input channels. 
        """
        self.tg.cfg_var['in_channels']=[1,7]
        self.tg.cfg_var['block_number'] = [0,2]
        self.tg.cfg_var['H'] += [301] # other than D & U, B block can also handle uneven W/H that are not multiples of 2 
        self.tg.cfg_var['W'] += [7] 
        cnt = 0
        for i in self.tg.generate_tests:
            for j in [True,False]: # D & U blocks should work without additionally sampling layers
                if j: 
                    sampled_layers = [(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))]
                else: 
                    sampled_layer = []
                with self.subTest('Testing:', block_type=block_type,i=i,sampled_layers=sampled_layers):
                    in_tensor = EncDecTestGenerator.get_ingoing_tensor(i)
                    i['out_channels'] = i['in_channels']*2
                    net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],sampled_layers,i)
                    out_tensor = net(in_tensor)
                    self.assertEqual(in_tensor.size()[2]*2, out_tensor.size()[2]) 
                    self.assertEqual(in_tensor.size()[3]*2, out_tensor.size()[3])
                    self.assertEqual(in_tensor.size()[1]*2,out_tensor.size()[1]) 
                    cnt += 1
        print(f'Tested {cnt} combinations.') # 432





    def test_get_block_D_empty_layers(self, block_type='D'):        
        """
        Test correct behaviour for D block > 1 when no layers are specified: resolution gets halved, channels doubled. Should work or every even input resolution. 
        """
        self.tg.cfg_var['in_channels']=[5,64]
        self.tg.cfg_var['block_number'] = [1,7]
        self.tg.cfg_var['H'] += [28,300] # other than D & U, B block can also handle uneven W/H that are not multiples of 2 
        self.tg.cfg_var['W'] += [30] 
        cnt = 0
        for i in self.tg.generate_tests:
            sampled_layers = [(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))]
            with self.subTest('Testing:', block_type=block_type,i=i,sampled_layers=sampled_layers):
                in_tensor = EncDecTestGenerator.get_ingoing_tensor(i)
                i['out_channels'] = i['in_channels']*2
                net, arch = get_block(block_type,i['block_number'],i['in_channels'],i['out_channels'],sampled_layers,i)
                out_tensor = net(in_tensor)
                self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]*2) 
                self.assertEqual(in_tensor.size()[3],out_tensor.size()[3]*2)
                self.assertEqual(in_tensor.size()[2],out_tensor.size()[2]*2) 
                self.assertEqual(in_tensor.size()[1]*2,out_tensor.size()[1]) 
                cnt += 1
        print(f'Tested {cnt} combinations.') 



if __name__ == '__main__':
    unittest.main()
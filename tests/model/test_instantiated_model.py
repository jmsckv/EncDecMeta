from model.instantiate_searched_model import EncDec
from utils.test_generator import EncDecTestGenerator
from utils.sampling import sample_config, validate_config
import unittest
import random
import torch



class TestEncDec(unittest.TestCase):
    def setUp(self):
        self.tg = EncDecTestGenerator()
        self.tg.cfg_var['dropout_ratio'] = (0.1, 0.4)
        self.tg.cfg_var['momentum_bn'] =  (0, 1)
        self.tg.cfg_var['lr'] =  [0.1, 0.01] 
        self.tg.cfg_var['momentum'] = (0.1, 0.99)
        self.tg.cfg_fixed['H'] = 2
        self.tg.cfg_fixed['W'] = 2
        
    def test_valid_output_shape(self):
        for i in [0,1,4]:
            for j in self.tg.generate_tests:
                n_blocks = 2
                for k in ['D_blocks', 'B_blocks','U_blocks']:
                    j[k] = [[(random.choice('OHCV'),random.randint(1,10))for i in range(random.randint(1,4))] for k in range(n_blocks)]
                j['H'] = j['H'] **(i+n_blocks)
                j['W'] = j['W'] **(i+n_blocks)
                j = sample_config(validate_config(j))
                net = EncDec(j)
                input_tensor = EncDecTestGenerator.get_input_tensor(j)
                out = net(input_tensor)
                with self.subTest('Evaluating sampled config:', j=j):
                    self.assertEqual(out.shape[2:], EncDecTestGenerator.get_output_tensor(j).shape[2:])
                    self.assertEqual(out.shape[1], j['classes'])
                    self.assertEqual(out.shape[0], j['batch_size'])
                    self.assertEqual(input_tensor.shape[0], j['batch_size'])


            


if __name__=='__main__':
    unittest.main()



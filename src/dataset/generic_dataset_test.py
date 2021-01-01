import unittest
from code.dataset.generic_dataset import GenericDataset
from code.configurations.chargrid import config
from torch.utils.data import DataLoader
import os
import random

class TestGenericDataset(unittest.TestCase):

    def test_deterministically_load_1_sample(self):
        """Test whether we deterministically select the same sample if we restrict datasets to 1 sample."""
        #ds = random.choice(os.listdir(os.environ['DATAPATH']))
        t = 0
        for i in ['train', 'val']: # for Cityscapes we don't have test labels, hence omitting test
            sampled = []
            ds = random.choice(['Cityscapes','Chargrid'])
            config['dataset']['selected'] = ds
            config['dataset'][config['dataset']['selected']]['samples_' + i] = 1
            d = GenericDataset(i, config)
            dl = DataLoader(dataset=d, batch_size=config['dataset'][config['dataset']['selected']]['batch_size_'+i], shuffle=True, drop_last=False)
            for j in range(random.choice(range(1,20))):
                for k in dl:
                    sampled.append(k)
            for i in range(len(sampled)-1):
                for j in [0,1]:
                    r = sampled[i][j] - sampled[i+1][j]
                    t += r.sum().item()
        self.assertAlmostEqual(t,0)




if __name__ == '__main__':
    unittest.main()

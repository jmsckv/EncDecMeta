from utils.sampling import sample_randomly, sample_block, sample_blocks, sample_arch, validate_and_sample_config
from utils.test_generator import EncDecTestGenerator
import unittest

class SampleRandomly(unittest.TestCase):
    """
    Test functionality of random sampling being applied to config file.
    """

    def test_sample_range(self):
        self.assertIn(sample_randomly(range(1,10)),range(1,10))
        self.assertEqual(sample_randomly(range(10,11)),10)
        self.assertNotIn(sample_randomly(range(5,10)),range(1,5))

    def test_sample_list(self):
        self.assertIn(sample_randomly(['A','B',2, None]),['A','B', 2, None])
        self.assertIn(sample_randomly(['A','B']),'ABC')
        self.assertNotIn(sample_randomly(['A','B']),'DEF')
        self.assertEqual(sample_randomly(['A','A']),'A')


    def test_sample_tuple(self):
        self.assertEqual(sample_randomly((1,1)),1)
        self.assertEqual(sample_randomly((.5,.5)),.5)
        self.assertGreater(sample_randomly((.5,.51)),.5)
        self.assertAlmostEqual(sample_randomly((0.999999999999,1)),1)
        with self.assertRaises(AssertionError):
            sample_randomly((1,))
            sample_randomly((1,2,3))
            sample_randomly((1,'2'))


class SampleArch(unittest.TestCase):
    """
    Validate and sample architecture blocks.
    """

    def test_sample_blocks(self):

        # deterministic
        b = [('H',3),('C',700),('V',2), ('O',1)] # 4 layers in 1 block
        self.assertEqual(sample_block(b),b) 
        self.assertEqual(sample_block(b*3),b*3) # still 1 block, now 4*3=12 layers
        self.assertEqual(sample_blocks([b]*3),[b]*3) # now 3 blocks with 4 layers
        # deterministic sampling
        bb = [(['H','H'],3),('C',range(700,701)),('V',[2,2]), (['O','O','O'],1)] # 4 layers in 1 block
        self.assertEqual(sample_block(bb),b) 
        self.assertEqual(sample_block(bb*3),b*3) # still 1 block, now 4*3=12 layers
        self.assertEqual(sample_blocks([bb]*3),[b]*3) # now 3 blocks with 4 layers
        # non-deterministic sampling
        b = [(['H','O'],[1]),('C',[2,2]),('V',range(3,10)), (['O','V','O'],1)] # 4 layers in 1 block
        sampled = sample_block(b)
        self.assertIn(sampled[0][0],'OH')
        self.assertEqual(sampled[0][1],1)
        self.assertEqual(sampled[1][0],'C')
        self.assertEqual(sampled[1][1],2)
        self.assertEqual(sampled[2][0],'V')
        self.assertLessEqual(sampled[2][1],10)
        self.assertIn(sampled[3][0],'OV')
        self.assertEqual(sampled[3][1],1)

        # catching assertion errors
        errors = []
        errors += [[('A',3),('C',700)]] # unknown str
        errors += [[(['A','H'],3),('C',700)]] # unknown str
        errors += [[('C',7.)]] # float
        errors += [[('H',3),('C',700),['O',1]]] # list instead of tuple
        errors += [()]
        for e in errors:
            with self.subTest('Failing for:', e=e):
                with self.assertRaises(AssertionError):
                    sample_block(e)   




    def test_sample_arch(self):
  
        # test case 1: fixed arch
        bl = [('H',3),('C',700),('V',2), ('O',1)] # 4 layers in 1 block
        cfg = {}
        a = bl * 2 
        b = bl 
        c = bl * 3 
        cfg['U_blocks'] = [a,[],a] 
        cfg['B_blocks'] = [b]
        cfg['D_blocks'] = [c] * 3 
        cfg = sample_arch(cfg)
        self.assertEqual(cfg['U_blocks'], [a,[],a]) 
        self.assertEqual(cfg['B_blocks'],[b]) 
        self.assertEqual(cfg['D_blocks'],[c]* 3) 


        # test case 2: deterministic sampling
        bl_s = [(['H','H','H'],3),('C',[700,700,700])]
        bl = [('H',3),('C',700)] 
        cfg = {}
        cfg['U_blocks'] = [bl_s,[],bl_s] 
        cfg['B_blocks'] = [bl_s]
        cfg['D_blocks'] = [bl_s] * 3 
        cfg = sample_arch(cfg)
        self.assertEqual(cfg['U_blocks'], [bl,[],bl]) 
        self.assertEqual(cfg['B_blocks'],[bl]) 
        self.assertEqual(cfg['D_blocks'],3*[bl]) 
        with self.assertRaises(AssertionError):
            cfg['B_blocks'] = []
            cfg = sample_arch(cfg)
        with self.assertRaises(AssertionError):
            cfg['B_blocks'] = [bl_s]
            cfg['D_blocks'] = [bl_s] * 2 
            cfg = sample_arch(cfg)



        # test case 3: non-deterministic sampling
        bl_s = [(['H','C','C'],range(2,3)),('V',[700,701])]
        cfg = {}
        cfg['U_blocks'] = [bl_s,bl_s,[]] 
        cfg['B_blocks'] = [bl_s,bl_s]
        cfg['D_blocks'] = [[],[],[]] 
        cfg = sample_arch(cfg)
        self.assertEqual(cfg['D_blocks'],3*[[]]) 
        for p in ['U_blocks','B_blocks']:
            for b in [0,1]:
                for i,l in enumerate(cfg[p][b]):
                    with self.subTest('Testing', p=p, b=b, l=l):
                        if i == 0:
                            self.assertIn(l[0],'HC')
                            self.assertEqual(l[1],2)
                        if i == 1:
                            self.assertEqual(l[0],'V')
                            self.assertIn(l[1],[700,701])



class SampleArchAndHps(unittest.TestCase):
    """
    Validate and sample architecture blocks.
    """

    def test_validate_and_sample_config_0(self):
        tg = EncDecTestGenerator()
        # generate valid test cases: multiples of 2
        tg.cfg_fixed['H'] = 4
        tg.cfg_fixed['W'] = 4
        c = [('C',2)] 
        tg.cfg_fixed['B_blocks'] = [c]
        for i in range(1,10):
            tg.cfg_fixed['D_blocks'] = [c]*i
            tg.cfg_fixed['U_blocks'] = [[]]*i
            tg.cfg_fixed['H'] = int(2*2**i)
            tg.cfg_fixed['W'] = int(2*2**i)
            for j in tg.generate_tests:
                with self.subTest('Evaluating for:', j=j):
                    cfg = validate_and_sample_config(j)
                    self.assertTrue(cfg)


    def test_validate_and_sample_config_1(self):
        # should raise assertion error
        with self.assertRaises(AssertionError): # change to ValueError > works
            tg = EncDecTestGenerator()
            c = [('C',2)] 
            tg.cfg_fixed['D_blocks'] = [c]*2
            tg.cfg_fixed['U_blocks'] = [[]]*2
            tg.cfg_fixed['B_blocks'] = [c]
            tg.cfg_fixed['H'] = 6
            tg.cfg_fixed['W'] = 16
            validate_and_sample_config(next(tg.generate_tests))




if __name__ == '__main__':
    unittest.main()



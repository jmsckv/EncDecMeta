

# TODO: unused
# idea is to replace the assertions included in the model/search pipeline/components with a check once before any model is instantiated.


class ConfigValidator():
    """
    A class that validates the specifcation of a model or search space in a configuration file.

    """

    def __init__(self,):

        super().__init__()


    
def check_valid_input_res(config):
    for r in ['height', 'width']:
        res = config[r]
        for i in range(len(config['downsampling_blocks'])):
            assert res % 2 == 0 , f'Specified number of downsampling layers results odd resulution: {r} in block {i}.'
            res /= 2


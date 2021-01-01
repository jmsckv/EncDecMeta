import sys
import os
import importlib


def parse_config_file(path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    assert path[-3:] == '.py', 'Config file must be a valid Python module and end with ".py".'
    module_name = path.split(os.sep)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    config = module.config
    assert isinstance(config,dict), "Config must be specified as a valid Python dictionary."
    return config










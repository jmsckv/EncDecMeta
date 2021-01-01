from pathlib import Path

import \
    yaml  # https://www.exploit-db.com/docs/english/47655-yaml-deserialization-attack-in-python.pdf?utm_source=dlvr.it&utm_medium=twitter


def dict_to_yaml(d: dict, file_path, overwrite_existing=False):
    if not overwrite_existing:
        assert Path(file_path).exists() is False, 'Outfile already exists, won\'t overwrite'
    with open(file_path, 'w') as file:
        yaml.dump(d, file)


def append_to_yaml(d: dict, file_path):
    if not Path(file_path).exists():
        Path(file_path).touch()
    with open(file_path, 'a') as file:
        yaml.dump(d, file)


def dict_from_yaml(file_path):
    with open(file_path, 'r') as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    return d


"""
def extend_yaml(path_to_yml: str, new_d: dict, return_new_d: bool=True):
    try:
        d = dict_from_yaml(path_to_yml)
    except FileNotFoundError as e:
        print('The specified path does not yield an existing YAML file, will create one now.')
        d = dict()
    if d is None:
        d = dict()
    if len(d)>0:
        for k in new_d:
            if k in d:
                assert d[k] == new_d[k], f'Conflict for key {k}, and between value {d[k],new_d[k]} in old and new dict, respectively. '
    d_out = {**d, **new_d}
    path_out, fn_out = os.path.split(path_to_yml)
    dict_to_yaml(d_out,fn=fn_out,out_dir=path_out)
    if return_new_d:
        return d_out

my_dict= {'a':1,'b':2}
dict_to_yaml(my_dict, './test.yaml')
d = dict_from_yaml('./test.yaml')
print(d)
"""

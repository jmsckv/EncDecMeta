from pathlib import Path
import os

def get_paths_for_dataset(dataset_name, create_non_existing=False, return_dict=True):
    p = Path(os.environ['DATAPATH'])
    p = p/dataset_name
    d = {}
    for i in ['proc']: # could add e.g. proc1,proc2,... if different preprocessing schemes
        d[i] = {}
        pp = p/i
        for j in ['data','labels']:
            ppp = pp/j
            d[i][j] = {}
            for k in ['train','val','test']:
                pppp = ppp/k
                d[i][j][k] = pppp
                if create_non_existing:
                    try:
                        pppp.mkdir(parents=True, exist_ok=False)
                    except:
                        print(f'Did not create {pppp} since already existing.')
    if return_dict:
        return d


#d = get_paths_for_dataset('TestDataSet')
#print(d)
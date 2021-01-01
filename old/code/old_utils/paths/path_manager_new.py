import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])

from old_utils.utils.serialization import dict_to_yaml, dict_from_yaml

from pathlib import Path
from filelock import FileLock
import yaml


class PathManager:
    '''
    Generic path handler for DS projects.
    Expects three environment variables to be set for a project: $DATAPATH, $MODELSPATH, $RESULTSPATH.
    Use Cases:
    1. Prior to preprocessing a dataset_name: set up folder structure to store raw and preprocessed overfit_single_sample ('meta').
    2. Before starting to train a model: tell the model where to fetch the overfit_single_sample from and where to store its results.
    Design Goals:
    1. Never overwrite existing directories/files.
    2. Separate overfit_single_sample from model_search and results.
    3. Make results of several experiments easily retrievable:
    (a) by using coherent directory/file structure
    (b) by aggregating results in $RESULTSPATH
    '''

    def __init__(self, dataset_name: str = None, model_name: str = None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_ID = None  # if populated, corresponds to mapping ID:run/model/dataset in $RESULTSPATH/lookup_yaml
        # self.model_name = model_name
        self.paths = dict()
        # below will follow
        # self.model_ID = None

        # get environment variables
        for e in ['DATAPATH', 'RESULTSPATH']: #, 'MODELSPATH']:
            p = os.environ[e]
            if p is None:
                raise Exception(f'Please specify an environment variable {e}.')
            self.paths[e.lower()] = p
            if not Path(p).exists():
                Path(p).mkdir(parents=True)

        ####################
        # POPULATE $DATAPATH
        ####################

        if dataset_name:
            # 1. path for raw overfit_single_sample: put the overfit_single_sample here as received from website/customer/..
            self.paths['dataset_path'] = os.path.join(self.paths['datapath'], self.dataset_name)
            self.paths['raw_data'] = os.path.join(self.paths['dataset_path'], 'raw')
            if not Path(self.paths['raw_data']).exists():
                Path(self.paths['raw_data']).mkdir(parents=True)

            # 2. for "proc"(essed) overfit_single_sample: the predefined folder structure serves as an API to all upstream tasks
            self.paths['proc_data'] = os.path.join(self.paths['dataset_path'], 'proc')
            for i in [
                'proc']:  # placeholder: in the future we may have proc1/proc2/... i.e. different preprocessing for the same raw overfit_single_sample
                for j in ['overfit_single_sample', 'labels']:
                    for k in ['train', 'val', 'test']:
                        path = os.path.join(self.paths['dataset_path'], i, j, k)
                        if not Path(path).exists():
                            Path(path).mkdir(parents=True)
                        self.paths[j + '_' + k] = path



        #######################################
        # POPULATE $RESULTSPATH and $MODELSPATH
        #######################################

        # currently, $RESULTSPATH consists of single lookup.yaml file
        # if a lookup yaml does not exist, create one
        # if one exists, don't overwrite it
        self.paths['lookup_yaml'] = os.path.join(self.paths['resultspath'], 'lookup.yaml')
        if not Path(self.paths['lookup_yaml']).exists():
            Path(self.paths['lookup_yaml']).touch()

        # given we assigned a model ID, also create a corresponding model path
        # this path should be empty; this condition gets checked by last line: exist_ok=False
        if self.model_name is not None:
            self.create_lookup_entry()
            model_path = os.path.join(self.paths['modelspath'], str(self.model_ID))
            self.paths['model_dir'] = model_path
            chekpoints_path = os.path.join(model_path, 'checkpoints')
            self.paths['checkpoints_dir'] = chekpoints_path
            if not Path(chekpoints_path).exists():
                Path(chekpoints_path).mkdir(parents=True)



    def create_lookup_entry(self, update_model_ID=True):
        """  Create a new entry in the lookup file, i.e. assign a free/new modelID."""
        with FileLock(self.paths['lookup_yaml'] + '.lock'):
            current_d = dict_from_yaml(self.paths['lookup_yaml'])
            if current_d is None:
                current_d = dict()
                next_index = 0
            else:
                next_index = len(current_d)
                current_d[next_index] = dict()
                # list here everything you consider important to decide for an architecture
                # by default and to ensure comparability, you have to specify everything key beforehand:
                # update_lookup_entry() only accepts keys specified here
                current_d[next_index]['model_name'] = self.model_name
                current_d[next_index]['replicate_chargrid'] = False
                current_d[next_index]['dataset_name'] = self.dataset_name

                # note: currently we update this information only once, when the training has finished in network.closing_procedure()
                # this is to avoid blocking the shared lookup_yaml too much
                # the lookup main purpose is to provide a mapping from sampled arch string to an unique identifier (used e.g. as a folder name)
                # in general, we could continuously update the lookup_yaml file with the best found results so far
                # we could also think of replacing the lookup file with a database
                current_d[next_index]['finished_training'] = False
                current_d[next_index]['best_val'] = None
                current_d[next_index]['best_val_in_epoch'] = None
                current_d[next_index]['max_epochs'] = None

                # a valid experiment comprises both, specification of dataset and model
                # later we can use this as a filter in pd df to filter out test runs
                if self.dataset_name and self.model_name:
                    current_d[next_index]['valid_experiment_specification'] = True
            dict_to_yaml(current_d, self.paths['lookup_yaml'], overwrite_existing=True)
        if update_model_ID:
            self.model_ID = next_index

    @staticmethod
    def update_lookup_entry(ID, key, value):
        """ Currently, we only update the information that a model has finished training."""
        path = os.path.join(os.environ['RESULTSPATH'], 'lookup.yaml')
        with FileLock(path + '.lock'):
            with open(path, 'r') as file:
                current_d = yaml.safe_load(file)
                assert current_d is not None, 'Can only update a populated lookup YAML.'
                assert ID in current_d.keys(), 'Can only update existing ID.'
                assert key in current_d[ID].keys(), 'Can only update existing key.'
                current_d[ID][key] = value
            with open(path, 'w') as file:
                yaml.safe_dump(current_d, file)


"""
# tests - only model or dataset, both, or neither provided
for i in range(3):
    i = str(i)
    pm = PathManager(dataset_name='D'+i,model_name='M'+i, print_summary= False)
    #pm = PathManager(model_name='M'+i, print_summary= False)
    #pm = PathManager(dataset_name='D'+i, print_summary= False)
    pm = PathManager()
    print(pm.paths)
PathManager.update_lookup_entry(1,'best_val','some_crazy_best_val_result')
PathManager.update_lookup_entry(0,'finished_training','first model has finished training')
"""

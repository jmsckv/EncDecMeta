import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])

from old_utils.paths.path_creation import check_for_existing_artifact_name

from filelock import FileLock
import yaml
import pprint


class PathManager:
    '''
    Generic path handler for DS projects.
    Expects three environment variables to be set for a project: $DATAPATH, $MODELSPATH, $RESULTSPATH.
    Use Cases:
    1. Prior to preprocessing a dataset_name: set up folder structure to store raw and preprocessed overfit_single_sample ('meta').
    2. Before starting to train a model: tell the model where to fetch the overfit_single_sample from and where to store its results.
    Design Goals:
    1. Never overwrite existing directories/files.
    2. Separate overfit_single_sample from model_search, and results.
    3. Make results of several experiments easily retrievable:
    (a) by using coherent directory/file structure
    (b) by aggregating results in $RESULTSPATH/shared and $RESULTSPATH/aggregated
    '''

    def __init__(self, dataset_name: str, model_name: str = None, print_summary: bool = False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.paths = dict()
        # below will follow
        self.model_ID = None

        # get environment variables
        for e in ['DATAPATH', 'RESULTSPATH', 'MODELSPATH']:
            path = os.environ[e]
            if path is None:
                raise Exception(f'Please specify an environment variable {e}.')
            self.paths[e.lower()] = path
            if not os.path.exists(path):
                os.makedirs(path)

        ####################
        # POPULATE $DATAPATH
        ####################

        # 1. path for raw overfit_single_sample: put the overfit_single_sample here as received from website/customer/..
        self.paths['dataset_path'] = os.path.join(self.paths['datapath'], self.dataset_name)
        if print_summary: print(self.paths['dataset_path'])
        self.paths['raw_data'] = os.path.join(self.paths['dataset_path'], 'raw')
        if not os.path.exists(self.paths['raw_data']):
            os.makedirs(self.paths['raw_data'])

        # 2. for "proc"(essed) overfit_single_sample: the predefined folder structure serves as an API to all upstream tasks
        self.paths['proc_data'] = os.path.join(self.paths['dataset_path'], 'meta')
        for i in [
            'meta']:  # placeholder: in the future we may have meta1/meta2/..., i.e. different preprocessing for the same raw overfit_single_sample
            for j in ['overfit_single_sample', 'labels']:
                for k in ['train', 'val', 'test']:
                    path = os.path.join(self.paths['dataset_path'], i, j, k)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    self.paths[j + '_' + k] = path

        #######################
        # POPULATE $RESULTSPATH
        #######################

        # logic/expected output:
        # $RESULTSPATH/datasetA/hashed_modelA.0
        #                      /hashed_modelB.0
        #                      /hashed_modelB.1
        #                      /hashed_modelB.2
        # $RESULTSPATH/datasetB/hashed_modelA.0
        #                      /...
        # interpretation: e.g. model B was trained 3 times on dataset_name B
        # note: here we don't want to store model weights but only epoch-wise performance statistics on val/train etc.
        # a script will parse the (sub)folders' YAML files and aggregate the results in a Pandas dataframe

        # 3a) create shared results folder: "single source of truth"
        # here we can easily look up the condensed performance/results of any model on a given dataset
        path = os.path.join(self.paths['resultspath'], 'aggregated', self.dataset_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.paths['aggregated_results'] = path
        # and also create a placeholder file to look abbreviated model names up
        path = os.path.join(self.paths['resultspath'], 'shared')
        if not os.path.exists(path):
            os.makedirs(path)

        self.paths['model_lookup_file'] = os.path.join(path, 'lookup.yaml')
        # 3b-optional)
        # applies if initialized with attribute model_name
        # because of file name length restrictions, we create a unique model ID and keep of it track in a locked file
        if self.model_name is not None:
            # step 1: get a unique model ID and serialize mapping from ID to model name
            with FileLock(self.paths[
                              'model_lookup_file'] + '.lock'):  # https://stackoverflow.com/questions/489861/locking-a-file-in-python
                if os.path.exists(self.paths['model_lookup_file']):
                    with open(self.paths['model_lookup_file'],
                              'r') as file:  # dict_from_yaml function does not work for some reason
                        current_d = yaml.safe_load(file)
                        if current_d is not None:
                            if self.model_name not in current_d.values():
                                next_index = len(current_d)
                                current_d[next_index] = self.model_name
                                with open(self.paths['model_lookup_file'], 'w') as file:
                                    yaml.safe_dump(current_d, file)
                                    self.model_ID = next_index
                            else:
                                self.model_ID = dict(zip(current_d.values(), current_d.keys()))[self.model_name]
                else:
                    current_d = dict()
                    current_d[0] = model_name
                    extend_yaml(self.paths['model_lookup_file'], current_d)
                    self.model_ID = 0

            # step 2: create folder named as model ID + identifier in case of repeated training runs of a model
            path, folder_name = check_for_existing_artifact_name(self.paths['aggregated_results'], self.model_ID,
                                                                 propose_alternative_if_necessary=True)
            path = os.path.join(path, folder_name)
            os.makedirs(path)
            self.paths['aggregated_results'] = path  # update this pointer in case we provide a model

            #################################
            # POPULATE $MODELSPATH (OPTIONAL)
            #################################

            # logic:
            # a model is always trained on one dataset_name (if you'd intend to train it on multiple: comprise them as one dataset_name)
            # a model can be trained several times on the same datset (e.g. different hyperparameters, seeds, etc.)
            # expected output:
            # $MODELSPATH/model0/datasetA.0/checkpoints
            #                              /info.yaml
            #                              /all_results.yaml
            # $MODELSPATH/model1/datasetA.0/checkpoints
            #                              /info.yaml
            #                              /all_results.yaml
            # $MODELSPATH/model1/datasetB.0/checkpoints
            #                              /info.yaml
            #                              /all_results.yaml
            # $MODELSPATH/model1/datasetB.1/checkpoints
            #                              /info.yaml
            #                              /all_results.yaml
            # intpretation: model0 was trained once on A; model1 was trained once on A and twice on B

            # 4a) for every training run of a model on a dataset_name, we create a new folder
            if self.model_name:
                path = os.path.join(self.paths['modelspath'], str(self.model_ID))
                if not os.path.exists(path):
                    os.makedirs(path)
                path, folder_name = check_for_existing_artifact_name(path, self.dataset_name,
                                                                     propose_alternative_if_necessary=True)
                path = os.path.join(path, folder_name)
                os.makedirs(path)
                # within this folder, we store information about the model and its results in YAML files
                self.paths['info_yaml_path'] = os.path.join(path,
                                                            'info.yaml')  # TODO: extend info.yaml with d dict during training
                open(self.paths['info_yaml_path'], 'a').close()
                # self.paths['all_results_yaml_path'] = os.path.join(path,'all_results.yaml')
                # open(self.paths['all_results_yaml_path'],'a').close()
            # 4b) we keep track of best model (weights), and latest model weights (to possibly resume training at later point in time)
            for i in ['checkpoints']:
                for j in ['best_result', 'latest_result']:
                    subpath = os.path.join(path, i, j)
                    if not os.path.exists(subpath):
                        os.makedirs(subpath)
                    self.paths[j + '_path'] = subpath

        # keep explicitly track whether we're repeatedly training a model (on the same dataset)
        self.n_runs = self.paths['aggregated_results'].split('.')[-1]

        if print_summary:
            print(f'Created and/or loaded path structure for:')
            print(f'\tDataset: {self.dataset_name}')
            if self.model_name:
                print(f'\tOriginal Model Name: {self.model_name}')
                print(f'\tAssigned Model ID: {self.model_ID}')
                print(f'\tRun {self.n_runs}')

            print(f'Paths:')
            pprint.pprint(self.paths)

# test
# pm = PathManager(dataset_name='Dataset2',model_name='Model123', print_summary= True)
# pm = PathManager(dataset_name='Dataset2', print_summary= True)

import os
import sys

sys.path.extend(os.environ['PYTHONPATH'])
import argparse
import pprint
from old_utils.paths.path_creation import create_core_routine


def default_paths_from_dataset_name(dataset_name, return_paths=True):
    default_paths = {}
    # assert env variables are set
    for e in ['DATAPATH', 'RESULTSPATH', 'MODELSPATH']:
        path = os.environ[e]
        if path is None:
            raise Exception(f'Please specify an environment variable {e}.')
        default_paths[e.lower()] = path

    # paths for raw and preprocessed overfit_single_sample
    # default_paths['dataset_name'] = dataset_name
    default_paths['dataset_name'] = create_core_routine(default_paths['datapath'], dataset_name,
                                                        propose_alternative=False)
    default_paths['raw'] = create_core_routine(default_paths['dataset_name'], 'raw')
    default_paths['meta'] = create_core_routine(default_paths['dataset_name'], 'meta')
    for i in ['meta']:  # previously also applied to 'raw', currently removed from list
        for j in ['overfit_single_sample', 'labels']:
            for k in ['train', 'val', 'test']:
                tmp_path = create_core_routine(default_paths[i], j)
                default_paths[j + '_' + k] = create_core_routine(tmp_path, k)
    # currently not used: put preprocessing and old_evaluation procedures here because they might be reused by different dataset
    # default_paths['procedures'] = create_core_routine(default_paths['dataset_name'],'procedures')
    # default_paths['raw_to_meta'] = create_core_routine(default_paths['procedures'],'raw_to_meta')
    # default_paths['old_evaluation'] = create_core_routine(default_paths['procedures'],'old_evaluation')

    # old_utils and aggregated results ctd. : overfit_single_sample/results/aggregated/results_aggregated , /training_on, ...
    default_paths['shared_results'] = create_core_routine(default_paths['resultspath'], 'shared_results')
    # default_paths['results_aggregated'] = create_core_routine(default_paths['resultspath'],'aggregated')
    # default_paths['training_on'] = create_core_routine(default_paths['results_aggregated'],'training_on')
    # default_paths['training_off'] = create_core_routine(default_paths['results_aggregated'],'training_off')
    # default_paths['finished_training'] = create_core_routine(default_paths['results_aggregated'],'finished_training')

    if return_paths:
        return default_paths


def get_paths_for_datasets(l, return_paths=True, verbose=True):
    if isinstance(l, str):
        l = [l]
    paths_dict = {d: default_paths_from_dataset_name(d) for d in l}  # dataset_name to paths keywords to abs paths
    if verbose:
        for k in paths_dict:
            print(f'\nCreated for dataset_name {k} these paths:')
            pprint.pprint(paths_dict[k])
    if return_paths:
        return paths_dict


def get_default_paths_for_model_and_dataset(dataset_name, model_name, return_paths=True, model_active=True):
    # logic 1: one model on one dataset_name at a given point in time
    # logic 2: if we train same model repeatedly, we will get different directories: model, model___1, model___2,...

    # copy default_op paths
    default_paths_dict = get_paths_for_datasets(dataset_name, return_paths=True, verbose=False)
    model_paths = dict(**default_paths_dict)
    # but only keep those paths actually needed

    # extend and update $MODELSPATH

    model_paths['model_name'] = model_name
    model_paths['model'] = create_core_routine(model_paths['modelspath'], model_paths['model_name'])
    model_paths['model'] = create_core_routine(model_paths['model'], dataset_name, propose_alternative=True)
    # model_paths['dataset_name'] = model_paths['model'].split('/')[-1]# is different from the initial model name if the model already was instantiated before
    # print(model_paths['dataset_name'])
    model_paths['checkpoints'] = create_core_routine(model_paths['model'], 'checkpoints')
    model_paths['best_result'] = create_core_routine(model_paths['checkpoints'], 'best_result')
    model_paths['latest_result'] = create_core_routine(model_paths['checkpoints'], 'latest_result')
    # model_paths['logs'] = create_core_routine(model_paths['model'], 'logs')
    model_paths['info'] = create_core_routine(model_paths['model'], 'info')

    model_paths['shared_results'] = create_core_routine(model_paths['shared_results'],
                                                        dataset_name)  # also store results on this level
    model_paths['shared_results'] = create_core_routine(model_paths['shared_results'], model_paths['model_name'],
                                                        propose_alternative=True)  # also store results on this level

    # TODO: register model as either active ('on') or inactive ('off'/'finished_training)
    # TODO: keep track of model results in results/aggregated

    if return_paths:
        return model_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create blank folder structure for given dataset_name.')
    parser.add_argument('--scope', help='enter the name of the dataset_name')
    args = parser.parse_args()

    if args.scope == 'thesis':
        get_paths_for_datasets(['Cityscapes', 'Chargrid'])

    elif args.scope == 'demo':
        # note: in demo, we set both results and modelspath
        for m in ['NN', 'RF', 'XGB']:
            for d in ['DatasetA', 'DatasetB']:
                model_paths = get_default_paths_for_model_and_dataset(model_name=m, dataset_name=d)
    else:
        get_paths_for_datasets(args.scope)

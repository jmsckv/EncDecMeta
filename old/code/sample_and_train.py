import os
import sys
sys.path.extend(os.environ['PYTHONPATH'])

import argparse
import importlib
import logging
import pprint
from datetime import datetime

import torch.utils.data

from ray import tune
from ray.tune.trainable import TrainableUtil
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHA

from dataset.generic_dataset import GenericDataset
from model.core_routines.forward_backward_passes import forward_pass_train, forward_pass_val
from model.instantiate_searched_model import ChargridSearched

from utils.sampling import sample_flat_hp, sample_hp_with_kwargs
from utils.count_parameters import count_parameters
from utils.printing import fancy_print
from model.metrics.metrics_aggregator import MetricsAggregator

logger = logging.getLogger(__name__)

# workaround to fix OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# https://www.gitmemory.com/issue/dmlc/xgboost/1715/464511708
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Trainable(tune.Trainable):
    def _setup(self, config):

        print('cuda available', torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.train_dataset = GenericDataset('train', config)
        if self.config['debug']:
            self.val_dataset = self.train_dataset # memorize 1 sample in debug bode
        else:
            self.val_dataset = GenericDataset('val', config)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=int(config['dataset'][config['dataset']['selected']]['batch_size_train']),
                                                        shuffle=True, drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                      batch_size=int(config['dataset'][config['dataset']['selected']]['batch_size_val']),
                                                      shuffle=False, drop_last=False)
        self.network = ChargridSearched(config=config)
        self.optimizer = sample_hp_with_kwargs(self.config, 'optimizer', external_kwargs={'params':self.network.parameters()})
        self.loss_fn = sample_flat_hp(self.config,'loss')
        self.config['parameter_count'] = count_parameters(self.network, save_to_file=os.path.join(self.logdir, 'parameter_count.txt'))
        self.results = {} # will get dynamically overwritten in every training iteration; also keeps track of best and last performance
        self.metrics_agg_train = MetricsAggregator('train', self.config)
        self.metrics_agg_val = MetricsAggregator('val', self.config)


    def _train(self):
        if self.config['verbose'] is True or self.config['debug'] is True:
            fancy_print(f'Epoch {self.training_iteration}')
        result_train = forward_pass_train(self.train_loader, self.optimizer, self.network, self.loss_fn, self.metrics_agg_train, self.config)
        result_val = forward_pass_val(self.val_loader, self.network, self.loss_fn, self.metrics_agg_val, self.config)
        for i in [result_val, result_train]:
            for k in i:
                self.results[k] = i[k]
        self.results['updated_best'] = False
        if self.training_iteration == 0:
            self.results['best_'+ self.config['tune']['key_metric']] = self.results[self.config['tune']['key_metric']]
        elif self.results[self.config['tune']['key_metric']] > self.results['best_'+ self.config['tune']['key_metric']]:
            self.results['best_'+ self.config['tune']['key_metric']] = self.results[self.config['tune']['key_metric']]
            self.results['updated_best'] = True
            self.update_best_performing()
        if self.config['verbose'] is True or self.config['debug'] is True:
            pprint.pprint(self.results)
        return self.results

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
        torch.save(self.network.state_dict(), checkpoint_path)
        return checkpoint_dir


    def update_best_performing(self, checkpoint_dir=None):
        '''
        This function is an extension to Trainable.save(), which according to the source code must not be overwritten.
        The function helps to continuously serialize the weights with the best val performance.
        '''
        if not self.results['updated_best']:
            return
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.logdir, 'checkpoint_best')
        TrainableUtil.make_checkpoint_dir(checkpoint_dir)
        _ = self._save(checkpoint_dir)



"""
TODO: 
stop fast but poorly performing models based on some additional criterion, could be sth. like:
if best val performance below threshold X/Y/Z after 10/20/30 epochs, then stop model
this criterion has to be passed to tune.run()'s stop argument:

stop (dict | callable | :class:`Stopper`): Stopping criteria. If dict,
    the keys may be any field in the return result of 'train()',
    whichever is reached first. If function, it must take (trial_id,
    result) as arguments and return a boolean (True if trial should be
    stopped, False otherwise). This can also be a subclass of
    ``ray.tune.Stopper``, which allows users to implement
    custom experiment-wide stopping (i.e., stopping an entire Tune
    run based on some time constraint).

def stop_too_fast_too_poorly_performing():
    ...
    return Stopper
"""


def run_tune(config, return_analysis=False):
    logger.info(pprint.pformat(config, indent=4))
    # construct scheduler if searching
    if config['tune']['scheduler']:
        scheduler = eval(config['tune']['scheduler'])(**config['tune'][config['tune']['scheduler']])
    else:
        scheduler = None
    if config['verbose']:
        print(scheduler)

    analysis = tune.run(
    Trainable,
    verbose = 1,
    name = config['tune']['experiment_name'],
    local_dir = os.environ['RESULTSPATH'],
    resources_per_trial = {'cpu': 4, 'gpu': 1},
    scheduler = scheduler,
    stop = {'training_iteration': config['max_epochs'] },
    num_samples = config['tune']['num_samples'],
    config = config,
    checkpoint_score_attr= 'best_' + config['tune']['key_metric'], # best_mIoU_val > When omitted, did not note any difference. What's the purpose?
    checkpoint_at_end = config['tune']['save_last'])

    if return_analysis:
        return analysis

def parse_args():
    parser = argparse.ArgumentParser(
        description='Randomly sample an architecture from model_search space and train it on either Cityscapes or Chargrid Datasets.'
                    'As a default baseline, you can specify to train the architecture from the original Chargrid paper.')
    parser.add_argument('--experiment_name',
                        help='Every experiment must have assigned a name. A corresponding folder will be created containing all trials.',
                        required=True,
                        default=None,
                        type=str)
    parser.add_argument('--config',
                        help='Specify which config you want to use to sample architectures and hyperparameters.',
                        required=True,
                        default=None,
                        type=str)
    parser.add_argument('--dataset',
                        help='Specify either "Cityscapes" or "Chargrid" as a dataset to train on.',
                        required=True,
                        default='None',
                        type=str)
    parser.add_argument('--num_samples',
                        help='Either, defines the number of samples to draw when performing random model with early stopping (ASHA). '
                             'Or, defines the number of repetitions a single architecture will train, in case config["fixed_arch"] is set to true.',
                        required=False,
                        default=None,
                        type=int)
    parser.add_argument('--test_run',
                        help='Run experiment only for a few epochs, train and val samples.',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--debug',
                        help='Debug the model  by:'
                             '\n- training and validating on exactly 1 sampled model for 10000 steps'
                             '\n- printing out intermediary tensor shapes',
                        required=False,
                        default=False,
                        type=bool)
    parser.add_argument('--verbose',
                        help='Print out losses, metrics, tensor shapes etc.',
                        required=False,
                        default=None,
                        type=bool)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # don't change the order of args!

    # which config?
    config_path = os.path.join(os.environ['PYTHONPATH'],'configurations')
    assert args.config in os.listdir(config_path), 'The arch you want to train has to be specified in code/configurations/<HERE.py>'
    m = importlib.import_module('.' + args.config[:-3], package='configurations')
    assert isinstance(m.config, dict)
    config = m.config
    # if config defines a single arch, and not a search space, we have to adjust some settings
    if config['fixed_arch']:
        config['tune']['scheduler'] = None
        config['tune']['num_samples'] = 1
    else:
        config['tune']['scheduler'] = 'ASHA'

    # which dataset?
    assert args.dataset in ['Chargrid', 'Cityscapes'], 'Please be specific whether you want to train on "Cityscapes" or "Chargrid".'
    if args.dataset == 'Chargrid':
        config['dataset']['selected'] = 'Chargrid'
    else:
        config['dataset']['selected'] = 'Cityscapes'
    # depending on which dataset is chosen, we have to adjust some settings
    config['max_epochs'] = config['dataset'][config['dataset']['selected']]['max_epochs']
    if config['tune']['scheduler']:
        config['tune'][config['tune']['scheduler']]['max_t'] = config['max_epochs']

    # experiment name?
    assert isinstance(args.experiment_name, str), "Please name the experiment you're going to run."
    config['tune']['experiment_name'] = args.experiment_name + '_' + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

    # optional: overwrite default num_samples
    if args.num_samples:
        assert isinstance(args.num_samples,int) and args.num_samples >= 1, "Number of samples has to be a positive integer."
        config['tune']['num_samples'] = args.num_samples

    # optional: test run
    if args.test_run:
        args.test_run = [int(i) for i in args.test_run.split(',')]
        assert sum(isinstance(i, int) for i in args.test_run) == 4, \
            "\nA test run has to be specified by four integers, separated by a comma (e.g. 1,2,3,4):\n" \
            "\t1. The number of epochs.\n" \
            "\t2. The number of training samples the model sees. \n" \
            "\t3. The number of validation samples the model sees.\n" \
            "\t4. The number of sampled architectures, or if given a fixed architecture the number of repeated runs.\n"
        config['test_run'] = args.test_run
        config['max_epochs'] = args.test_run[0]
        if config['tune']['scheduler']:
            config['tune'][config['tune']['scheduler']]['max_t'] = config['max_epochs']
        config['dataset'][config['dataset']['selected']]['samples_train']  = args.test_run[1]
        config['dataset'][config['dataset']['selected']]['samples_val']  = args.test_run[2]
        config['tune']['num_samples']  = args.test_run[3]

    # optional: debug
    if args.debug is True:
        config['debug'] = True
        config['sanity_checks'] = True
        assert args.test_run is None, 'Command line args --test_run and --debug are not intended to work together, choose one.'
        config['max_epochs'] = 10000
        if config['tune']['scheduler']:
            config['tune'][config['tune']['scheduler']]['max_t'] = config['max_epochs']
        config['tune']['max_epochs'] = config['max_epochs']
        config['dataset'][config['dataset']['selected']]['samples_train'] = 1
        config['dataset'][config['dataset']['selected']]['samples_val'] = 1
        config['verbose'] = 2
        config['tune']['num_samples']  = 1

    if args.verbose:
        console_handler = logging.StreamHandler()
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            handlers=[console_handler])
        config['verbose'] = args.verbose
    else:
        console_handler = logging.StreamHandler()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            handlers=[console_handler])

    run_tune(config)

if __name__ == '__main__':
    main()

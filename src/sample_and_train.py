import argparse
import importlib
import logging
import pprint
import os 
import shutil
import glob
import sys
from datetime import datetime
from copy import deepcopy

import torch.utils.data
from torch.optim import SGD
from torch.nn import CrossEntropyLoss 

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHA

from dataset.generic_dataset import GenericDataset

from model.instantiate_searched_model import EncDec
from model.core_routines.forward_backward_passes import forward_pass_train, forward_pass_val

from utils.sampling import validate_config, sample_config

from utils.parse_config import parse_config_file
from utils.count_parameters import count_parameters
from utils.printing import fancy_print
from model.metrics.metrics_aggregator import MetricsAggregator

logger = logging.getLogger(__name__)

import random

class Trainable(tune.Trainable):
    """
    A minimum viable Trainable object as outlined here: https://github.com/ray-project/ray/blob/releases/ray-1.1.0/python/ray/tune/trainable.py
    We need to implement ``setup``, ``step``, ``save_checkpoint``, and ``load_checkpoint`` when subclassing Trainable.
    TODO: for ASHA do we need to implement load_checkpoint?
    TODO: implement cleanup method to kill the Ray actor process?
    """

    def setup(self, config):

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.config = config
        self.config['keep_best_only'] = not bool(self.config.get('checkpoint_freq', 0))

        self.train_dataset = GenericDataset('train', self.config)
        self.val_dataset =  GenericDataset('val', self.config) 
        
 
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size= int(self.config['batch_size']),
                                                        shuffle=True, drop_last=False)

        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                      batch_size= int(self.config.get('batch_size_val', self.config['batch_size'])),
                                                      shuffle=False, drop_last=False)

        self.network = EncDec(self.config)
        
        self.network.serialize_config(self.logdir)
        count_parameters(self.network)

        self.optimizer = SGD(self.network.parameters(),lr = self.network.config['lr'], momentum=self.network.config['momentum'], weight_decay=self.network.config['weight_decay'], nesterov=self.network.config['nesterov'])
        self.loss_fn =  CrossEntropyLoss(weight=None)

        self.results = {} # required by Tune: will get dynamically overwritten in every training iteration
        
        self.metrics_agg_train = MetricsAggregator('train', self.config)
        self.metrics_agg_val = MetricsAggregator('val', self.config)
        

    def step(self):

                
        if self.config.get('verbose', False):
            fancy_print(f'Epoch {self.training_iteration}.')

        # in the first epoch of any instantiated model, we always set debug to true
        # we will then check in many places for the correctness of tensor shapes and that tensors reside on the GPU if available
        # after the first iteration, we don't need these checks any longer      
        self.config['debug'] = 1 if self.training_iteration == 0 else 0

        result_train = forward_pass_train(self.train_loader, self.optimizer, self.network, self.loss_fn, self.metrics_agg_train, self.config)
        result_val = forward_pass_val(self.val_loader, self.network, self.loss_fn, self.metrics_agg_val, self.config)

        for i in [result_val, result_train]:
            for k in i:
                self.results[k] = i[k]

        self.results['updated_best'] = False

        # special case: epoch 0
        if self.training_iteration == 0:
            self.results['best_'+ self.config['key_metric']] = self.results[self.config['key_metric']]
            self.save()

        # update best result if applicable
        if self.results[self.config['key_metric']] > self.results['best_'+ self.config['key_metric']]:
            self.results['best_'+ self.config['key_metric']] = self.results[self.config['key_metric']]
            self.results['updated_best'] = True
            if self.config.get('verbose', False):
                print(f'Updated incumbent in epoch {self.training_iteration}.')
    
        # special case: only serialize best performing model (weights as in epoch when best result achieved)
        if self.results['updated_best']:
            self.save()
            if self.config['keep_best_only']:
                candidates = [c for c in  glob.glob('checkpoint*') if c[-1] != str(self.training_iteration)]
                if len(candidates): # there should be at most one other checkpoint, we have to delete/update
                    shutil.rmtree(candidates[0]) 
        
        if self.config.get('verbose', False):
            pprint.pprint(self.results)


        return self.results # gets continuously logged by Tune


    def save_checkpoint(self, checkpoint_path):
        # TODO: make model (de)serialization properly work: https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        # problem: parameters are grouped in nn.Module lists
        # torch.save(self.network.state_dict(), os.path.join(checkpoint_path,'model.pt'))
        # current solution relies on pickle
        torch.save(self.network,os.path.join(checkpoint_path,'model.pt'))
        return checkpoint_path


def run_tune(config, return_analysis=False):

    # TODO: setup non-Tune logging
    # logger.info(pprint.pformat(config, indent=4))    
    
     # construct scheduler if searching
    if config.get('num_samples', 1) > 1:
        scheduler = ASHA(time_attr='training_iteration',\
                        metric='best_'+ config.get('key_metric', 'mIoU_val'),\
                        mode = 'max',
                        grace_period = config.get('grace_period', 3),
                        max_t = config.get('max_t', 500),
                        reduction_factor= config.get('reduction_factor',3),
                        brackets=config.get('brackets',1))
    else:
        scheduler = None

    analysis = tune.run(
    Trainable,
    verbose = config.get('verbosity_tune',2),
    name = config['experiment_name'],
    local_dir = config.get('RESULTSPATH',os.environ['RESULTSPATH']),
    resources_per_trial = {'cpu': config.get('cpus_per_trial', 1), 'gpu': config.get('gpus_per_trial', 1 if torch.cuda.is_available() else 0)},
    scheduler = scheduler,
    stop = {'training_iteration': config.get('max_epochs', 500)},
    num_samples = config.get('num_samples',1),
    config = config,
    checkpoint_score_attr = 'best_' + config.get('key_metric', 'mIoU_val'),
    checkpoint_at_end = config.get('checkpoint_at_end', False),
    checkpoint_freq = config.get('checkpoint_freq', None),
    progress_reporter = tune.progress_reporter.CLIReporter(metric_columns=['best_' + config.get('key_metric', 'mIoU_val'),'training_iteration']),
    loggers = [tune.logger.CSVLogger,tune.logger.TBXLogger],
    # log_to_file = ('stdout.log', 'stderr.log') # TODO: implement non-Tune logging
    )

    # you can have the results returned in a df, e.g. if running from within Jupyter notebook
    if return_analysis:
        return analysis


if __name__ == '__main__':
    
    cfg = parse_config_file(path=sys.argv[1]) # we read-in dictionary from a path specified in sys.argv[1]

    # TODO: implement non-Tune logging
    #console_handler = logging.StreamHandler() )
    #logging.basicConfig(level= logging.DEBUG if cfg.get('debug', False) else logging.INFO,
    #                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    #                    handlers=[console_handler])
    
    cfg = validate_config(cfg)
    if cfg.get('verbose',False):
        fancy_print(f'Successfully validated the configuration file.')
    
    run_tune(cfg)


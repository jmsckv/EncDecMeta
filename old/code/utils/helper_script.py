import logging
logger = logging.getLogger(__name__)
import time

import os
import sys
sys.path.extend(os.environ['PYTHONPATH'])

import importlib
import argparse
from copy import deepcopy





def set_logging_config(fn_components:list=[],path=None,level='WARNING'):
    fn = '_'.join(fn_components) + '.log'
    if path:
        fn = os.path.join(path,fn)
    level = eval('logging.' + level)
    return logging.basicConfig(filename=fn,
                        level=level,  # change to INFO, see what happens with log file
                        format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s')





def parse_args():
    parser = argparse.ArgumentParser( description= 'Dynamically evaluated helper script - functionality depends on defined scope.')
    parser.add_argument('--scope',
                        help = 'Define which kind of helper script is going to be executed.',
                        required = True,
                        type = str)
    parser.add_argument('--dry_run',
                        help = 'Dry run to test whether all expected configuration files and datasets were found.',
                        required = False,
                        default = False,
                        type = bool)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # shared between scopes
    config_path = os.path.join(os.environ['PYTHONPATH'],'configurations')

    if args.scope == 'debug_all_fixed_archs':
        debug_path = os.path.join(os.environ['RESULTSPATH'],'debug')
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        s = 'dry_run' if args.dry_run else ''
        set_logging_config([args.scope, s, str(time.time())], path=debug_path, level='INFO')
        for d in ['Chargrid', 'Cityscapes']:
            for c in [i for i in os.listdir(config_path) if not i[0] in '_.']:
                logger.info(f'\nDebugging config {c} on dataset {d}.')
                if args.dry_run: logger.info('Dry run.')
                try:
                    config = None  # important, otherwise carrying over dict when try failsls

                    m = importlib.import_module('.' + c[:-3], package='configurations')
                    config = m.config
                    if ('fixed_arch' not in config.keys() or config['fixed_arch'] is False) and c != 'chargrid.py':
                        # we need this exception for the 'base config' in chargrid.py since config is not imported
                        logger.info(f'Config in {c} is not specified to cover a fixed arch, hence excluded.')
                        raise KeyError
                    exec_string = 'python $PYTHONPATH/sample_and_train.py'
                    exec_string += ' --experiment_name ' + 'auto_debug_' + d + '_' + c
                    exec_string += ' --dataset ' + d
                    exec_string += ' --config ' + c
                    exec_string += ' --debug True'
                    logger.info(f'Will execute: {exec_string}')
                    if not args.dry_run:
                        os.system(exec_string)
                    logger.info(f'Successfully debugged config {c} on dataset {d}.')
                except:
                    logger.exception(f'Failed to successfully debug config {c} on dataset {d}.')






    if args.scope == 'log_test':
        #logging.basicConfig(filename= 'test.log',
                      #      level=logging.INFO, # change to INFO, see what happens with log file
                       #     format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s')
        set_logging_config(['testA','testB'])
        logging.info('Great, we are learning to properly log!')
        try:
            print('In main loop, first everything works.')
            i = 1/0 # now something fails, we don't anticipate the type of error
            logging.info('This will not get logged!')
        except Exception:
            #logger.warning("Fatal error in main loop", exc_info=True)
            logger.exception("Fatal error in main loop")
            print('Finished.')






if __name__ == '__main__':
    main()







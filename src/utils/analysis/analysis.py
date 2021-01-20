import os
from ray.tune import Analysis
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy 
import time



# TODO: timing  seconds >> day hour min 
# TODO: print out: config
# TODO: print out: best three architectures found > separate txt file
# 

# TODO: store number of parameters in results_dict, hence make it available for analysis


def human_time(time):
    """Convert seconds to days,hours,minutes format."""
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return "%dd %dh %dm %ds" % (day, hour, minutes, seconds)




def get_pixel_counts_cityscapes():
    # TODO: calculate correlations > class performance depending on # of pixels in training set?
    # TODO: is in best models this relation weaker as
    pass

def round(i, dec=2):
    if isinstance(i, np.float64):
        return np.around(i,dec)
    else: 
        return i

def get_tune_dfs(path):
    """
    Here we load the df which is produced/logged by Tune.
    For more details see here:  https://docs.ray.io/en/latest/tune/api_docs/analysis.html#tune-analysis-docs
    """
    analysis = Analysis(path)
    d = analysis.trial_dataframes
    return d

def enrich_df(df):
    """
    We enrich the results of every single trial and condense result into a df with one column.
    For every search, these single column df will then be concatenated and one csv stored to disk.
    """
    # best validation performance and corresponding index
    best = df['best_mIoU_val'].max()
    i_best = df[df['best_mIoU_val']==best].index[0]

    # enrich with additional information
    # is mIoU really sufficient to look at?
    # let's look at other statistics of best found model per trial
    for i in ['val']: # test set not publicly available
        for j in ['std','median']:
            sel = [c for c in df.columns if 'IoU' in c and 'class' in c and i in c]
            df[j+'_epoch_best_mIoU_'+i] = eval('df[sel].iloc[i_best].' + j + '()')


    # epoch with best mIoU does not mean that for every class in this epoch the best IoU was observed
    # for every class we find out the best IoU and how much we are off in the model we decide for
    # could be interesting future research question: e.g. dynamic class weights focusing on class with relatively low IoU
    for col in [c for c in df.columns if 'IoU' in c and 'class' in c and 'val' in c]:
        best_per_class = df[col].max()
        df['delta_best_mIoU_best_' + col] = best_per_class - best 

    # how long did this trial train in total?
    # helps us to group in upstream tasks according to ASHA's rungs
    df['max_epochs_trained'] = df['training_iteration'].max()

    return df.iloc[[i_best]] 
        



def get_results(path, store_as_csv = True, console_print = True, summary_report=True):
    """
    Get results of either single arch or search. 
    Specify path to root direcory containing all Tune Trainables.
    Also verbosely rename columns such that they can be easily displayed in LaTex report.
    The convention is to upper-case the most relevant (printed) columns.
    """

    # Aggregate information at trial level.
    d = get_tune_dfs(path)
    dfs = []
    for k in d: 
        df = d[k]
        df = enrich_df(df)
        dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(drop=True,inplace=True) # TODO: there is still a redundant column 


    # Aggregate information of best observed trial.
    best = df['best_mIoU_val'].max()
    i_best = df[df['best_mIoU_val']==best].index[0]

    # Everything below returnd in LaTex report, hence uppercased + space.
    df['Best mIoU (val)'] = best 
    df['Training time best (epochs)'] = df.iloc[i_best]['training_iteration']
    df['Training time best (seconds)'] = human_time(df.iloc[i_best]['time_total_s'])
    
   # Delta statistics of best model.
    sel = [c for c in df.columns if 'delta' in c and 'IoU' in c and 'class' in c and 'val' in c]
    for j in ['mean','median','std', 'max', 'min']:
        df['Best model: mIoU vs. best IoU per class (' + j + ')'] = eval('df[sel].iloc[i_best].' + j + '()')


    # Aggregate information accross trials.
    # We aggregate by ASHA's decision gates = epochs when trials get stopped.
    c = Counter(df['max_epochs_trained'].values)
    keys = sorted([int(c) for c in c.keys()],reverse=True)
    for e in keys:
        df[f'Configurations trained for max. {str(e)} epochs'] = c[e]
        for o in ['mean','median','std', 'max', 'min']:
            df[o.capitalize() + ' best mIoU of configurations trained for ' + str(e) + ' epochs'] = \
            eval('df[df["max_epochs_trained"]==e]["best_mIoU_val"].' + o +'()')

    if store_as_csv:
        df.to_csv(os.path.join(path,'results_aggregated.csv'))
    
    # columns considered most relevant for summary report: 
    cols = [c for c in df.columns if c[0].isupper() and c[0] != 'I']
    
    if console_print:
        for c in cols: 
            if c.startswith('Configurations'):
                print('*'*80)
            print(c,':',round(df.iloc[i_best][c]))


    if summary_report:
        outfn = os.path.join(path,'results_summary.txt')
        with open (outfn, 'w') as wf:
            for c in cols: 
                if c.startswith('Configurations'):
                    wf.write('\n')
                wf.write(c +': '+ str(round(df.iloc[i_best][c])))
                wf.write('\n')
    

    return df



path = 'archived/fixed_archs/unet_bs5/run_0'
print('running')
df = get_results(path)
print(df.shape)


#all_dataframes = get_tune_dfs(path)
#print(all_dataframes)
print('finished')
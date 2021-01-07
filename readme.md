# A User-Friendly Encoder-Decoder Meta-Search-Space For Semantic Segmentation

This repo allows to easily specify and search encoder-decoder architectures and associated hyperparameters. It is based on PyTorch and Ray Tune, an the asysnchronous successive halving algorithm (ASHA) as a search strategy.


**The current key use case is automating building robust, searched baselines for semantic segmentation tasks.**
Current key restrictions are no data augmentation mechanisms and no ResNet-like or DenseNet-like connections between convolutional layers.


## Quickstart

1. Clone this repository: ```https://github.com/jmsckv/EncDecMeta.git && cd EncDecMeta ```
This code is tested with CUDA 10.2, Python 3.7.7, and setuptools 20.3.3 on Ubuntu 18.04. Higher versions should generally be supported.

2. We recommend to launch a Docker container with `. build_and_run_docker.sh` (use `_cpu.sh` if no GPU is available).  This will automatically create the expected directory structure and environment variables. It also auto-detects free ports for JupyterLab ($PORT1), Tensorboard ($PORT2), and the Ray Dashboard ($PORT3). Run `docker ps` to see where to retrieve e.g. JupyterLab in your browser, the default password, which you can change in `jupyter_notebook_config.py` before launching the container, is ASHA2020.

3. Create a Python virtutal env to install the project libraries. Do so from $CODEPATH in the Docker container, which maps to the root of this repo.
```
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install encdecmeta
# pip install -e . # run this instead the previous command to install in editable mode
```

4. Specify $PROC_DATAPATH which should map to the preprocessed data. Below, in the section Data Layout, we describe in depth the naming conventions we expect. In the Docker container this env variable is automatatically set. It maps $CODEPATH/data/proc within the container to EncDecMeta/data/proc on your local disk.

5. Specify $RESULTSPATH where any experimental results are being stored. In the Docker container this env variable is automatatically set. It maps $CODEPATH/results within the container to EncDecMeta/results on your local disk.

5. Run Experiments with `$CODEPATH/src/sample_and_train.py <YOUR_CONFIG.py>.` <YOUR_CONFIG.py> must be a .py file containing a dictionary named config. You can look at the Python files in `$CODEPATH/src/configurations/` to learn about specifying a configuration dictionary.

## Example: U-Net

We can define an architecture close to the U-net proposed by Ronneberger et al. (2015) as follows (see `src/configurations/unet.py`for more details).


```
config = {'experiment_name': 'unet_fixed',
'D_blocks': [[c],[c],[c],[c],[]],
'B_blocks': [[c]], 
'U_blocks': [[c],[c],[c],[c], [c,c]],
'H': 256, 
'W': 512,  
'dropout_ratio': 0.1,
'momentum': 0.99,
'momentum_bn': 0.1,
'lr': 0.01,
'weight_decay': 0.001,
'nesterov': False,
'base_channels': 64, 
'batch_size': 1,
'max_t': 500} #  max epochs (in ASHA's terms 'budget') any model may train; if searching, ASHA's early stopping points are derived from this quantity
```

Train this model with: `$CODEPATH/src/sample_and_train.py $CODEPATH/src/configurations/unet.py`


Instead of deciding for this fixed architecture, we can embed the above model in a search space (cf. `src/configurations/unet.py`) by altering the following the above dictionary as follows:

```
c = (['H','V','C','O'], range(1,8)) # sampled layer: sample operation and dilation rate > more below
config['experiment_name'] = 'unet_searched'
config['num_samples'] = 500 # evaluating 500 samples from this search space
config['dropout_ratio']: (0,0.5), # sample from interval > continuous hyperparameter
config['momentum']: (0.5,1), # 
config['momentum_bn']: (0,1), 
config['lr']: [i*j for i in [1,3,5,7] for j in [0.1, 0.01, 0.001]], # sample from list > discrete hyperparameter
config['weight_decay']: [i*j for i in [1,3,5,7] for j in [0.01, 0.001, 0.0001]],
config['nesterov']: [True,False],
config['base_channels']: range(32,65),  # sample from range > discrete hyperparameter
config['batch_size']: range(1,11),
```
We search jointly for a good configuration of the SGD optimizer, regularization, and architecture. Batch size and number of base channels can generally result in OOMs. In this case, simply another candidate will get sampled, no manual intervention is required.

In general, we sample uniformly at random and from either lists, range objects or tuples. (Note: naively, you could model other distributions through repeating elements in a list).

If a tuple contains first a list and then a list or range object it describes a convolutional layer.
In this case, we sample from both tuple entries independently: first a layer operation, second the dilation rate.
Currently, there are 4 operations supported: 
- 'H' (horizontal) maps to a 1x3 convolution
- 'V' (horizontal) maps to a 3x1 convolution
- 'C' (horizontal) to a 3x1 convolution

- 'O'


Let's explain the above specifivation of the searched convolutional layer in more detail: `c = (['H','V','C','O'], range(1,8))`.

This framework propose 











## Key Use Cases:
- automating building a robust baseline for encoder decoder architectures:
    - baselines reported  
- experimenting with 


# Read-Me


This framework allows to specify architectures analoguous to search spaces as




The key idea is to make any instantiated model agnostic of whether it is being sampled as part of a search process or as a stand-alone model.
As such the sampling and the model-related logic is in in pure PyTorch and only relies on Python's standard library, whereas the applied search alogrithm may 
be selected by the user. You can switch to the Ray branch of this repository to see an example of how EncDecMeta is being combined with the implementation of the ASHA search algorithm in Ray Tune.

The search space consists of three abstractions:

- `Downsampling Blocks` halving the resolution of incoming feature maps while doubling the number of channels. The first operation in such a block is always hard-coded to be a 3x3 convolution with stride 2. After this layer, an arbitrary number of layers can be specified within the block. See more on the candidate operations below. 

- `Bottleneck Blocks` keeping both the number of feature maps and the spatial resolution constant. Within these blocks an arbitrary number of layers can be specified.

- `Upsampling Blocks` always double the spatial resolution while halving the number of outgoing feature maps compared to the previous block. In these blocks the first two layers are hardcoded. Firstly, a 1x1 depthwise convolution fuses feature maps from the previous decoder block and the horizontally skip-connected encoder blocks of the same spatial resolution. Then a 3x3 transpose convolution with stride 2 guarantees the upsampling by a factor of two. Afterwards, an arbitrary number of layers can be horizontally stacked.


These operations can be selected when specifiying a block:







EncDecMeta fixes certain design choices, while keeping a high flexibility for other ones:





containing a conv
The idea is to abstract the instantiation of a model 


neural architecture search algorithm and itsimplementation are abstracted away from the

EncDecMeta has these primary use



, see the Ray branch of this repository for an example of applying the bandt

PyTorch-based,



It consists of stackable downsampling, bottleneck, and upsampling blocks, which can contain









The applied search strategy is random sampling combined with early stopping.
Search spaces are defined as dictionaries in a .py configuration file. Fixed architectures can be defined analoguously, think of them as search spaces with exactly one choice to sample from.
Under `/configurations`, a readme describes how to proceed and several example configuration files are provided.

In general, an experiment may yield two outcomes. Either single, fixed architectures (usually trained to convergence) or the results of a search process. In the latter case, at least one model will be trained to convergence and several other models will be partially trained due to early stopping.
In any case, for every stopped model two states are serialized: the model parameters after the last trained epoch as well as the model parameters of the epoch with the best validation performance.


In the remainder, we describe how to navigate the repository as well as how to replicate the experiments.




# Docker Setup, Paths

All the below commands assume working from within the same shell on Linux/Mac.

1. Clone this repository.

2. Change into root direcory of cloned repository: `cd EncDec; GITPATH=$(pwd)`.

3. Set environment variables and create folders before starting Docker container.
    
    Please adjust the paths in set_env_vars.sh, then run `source dependencies/set_env_vars.sh`
    
    - `$GITPATH` should map to the root folder of the cloned repository. Within the container this path is mapped to `/work/git`.  
    - `$RESULTSPATH` should map to a folder (preferably local SSD), where the outputs of any training/search get stored,  e.g. model weights and other artefacts. Within the container this path is mapped to `/work/results`.
    - `$DATAPATH` should map to a folder (preferably local SSD),  where the input of any training/search gets stored. Within the container this path is mapped to `/work/data`. Please see more below on how to preprocess the data.

All these environment variables will persist inside the Docker container, where `$GITPATH` is also referred to as the environment variable `$PYTHONPATH`.


4. Get Docker running:

    - build Docker image: `cd $GITPATH/dependencies; docker build -t $IMAGENAME .`
    - run Docker container: `source $GITPATH/dependencies/run_docker.sh` 

You an access JupyterLab know in your browser at `localhost:<PORT1>`. The default password is ASHA2020. You can change it in `dependencies/jupyter_notebook_config.py.

# Quickstart - How to run an experiment.

We assume that `$DATAPATH/<DATASET_NAME>` has already been populated. Below, it is further described what input data structure is being expected. 

The central function to launch any training or search is `code/sample_and_train.py`.
It can be used for different use cases depending on the file being passed to the `--config` argument: training a single architecture, searching over different architectures, searching over non-architectural hyperparameters, or a combination of the latter two.

By default, you have to to specify three arguments to execute `sample_and_train.py`: 
1. `--exeriment_name <EXP_NAME>`, which will create a directory  `results/EXP_NAME/` containing everything related to the launched experiment, e.g. intermediary results, logs, and serialized weights. Every experiment should be assigned a unique experiment name. By running `tensorboard --logdir $RESULTSPATH/EXP_NAME/ --bind-all` inside a terminal in JupyterLab, you can analyse all models and results being associated with an experiment (also during training).
2. `--dataset <DATASET_NAME>`, which refers to a previously created dataset, on which a model is trained or architectures are being searched. Within the scope of the thesis, <DATASET_NAME> is restricted to either `Chargrid` or `Cityscapes`. In general, the specified dataset is expected to reside in a predefined folder structure in `DATAPATH/DATASET_NAME`. See more below.
3. `--config <CONFIG_NAME>`, which references a Python configuration file `code/configurations/<CONFIG_NAME>`. The file is expected to contain a dictionary named `config` encoding the experiment to be run. Setting a flag `config['fixed_arch']=True` will always result and training exactly one (possibly sampled) architecture for one time (note: with `--num_samples` you cou can adjust this behaviour (see next). If this flag is not set or set to False, a tune scheduler will be contructed and trained on num samples (all to be defined in the nested dict in `config['tune']`.) **Specifying a config file in `code/configurations/<YOUR_CONFIG>` and passing `--config <YOUR_CONFIG>` to  `code/sample_and_train.py` is what allows you to define your own search space.** Also see `code/configurations/readme.md` for further advice and explanations on how to generate configuration files.

In order to replicate the architecture and hyperparameters setting from the original Chargrid, one would run:

`python $PYTHONPATH/sample_and_train.py --dataset_name Chargrid --config chargrid_10k.py --experiment_name replicate_chargrid_0`

To search for an architecture while keeping other hyperparameters fixed (cf. experiment cited in thesis), one would run:

`python $PYTHONPATH/sample_and_train.py --dataset_name Chargrid --config search_arch.py --experiment_name search_new_chargrid_0`


In addition to the default and mandatory command line arguments, further arguments may be specified.

4. `--num_samples <INT>`, which in the case of a fixed arch would lead to repeated training with different random seeds. In the case of a non-fixed arch this will imply that a certain number of architectures and/or hyperparameter combinations will get evaluated when running the search. This overwrites the default of 1000.

5. `--test_run <INT1,INT2,IN3,INT4>`, allows to overwrite certain default values in the selected configuration file. `<INT1>` specifies the max. number of epochs a single architecture may train, `<INT2>` the number of train samples, `<INT3>` the number of validation sampled, and `<INT4>` the number of samples.

6. `--verbose True`, will print out additional information to get familiarized with the library, e.g. intermediary events (instantiation of a data sampler), sampled hyperparameters,  tensor shapes during forward pass, epoch losses, ...

7. `--debug True`, which cannot be combined with `--test_run`, is intended for checking if a sampled model can overfit on a single example when seeing it 10000 times. The returned training and validation performance metrics should be equivalent, since both sets consist of the same single sample. By default, `--verbose` will be set to True in debug mode.

# Data - How to format your datasets.

The current pipeline only supports two data sets (Chargrid and Cityscapes), being preprocessed as described in `code/preprocessing/preprocessing_{Cityscapes,Chargrid}.ipynb`. In general, also other datasets may easily be supported, which would however require certain abstractions in the code base.
In general, RGB images are supported  as `.png` files. Images with more than 3 channels such as Chargrid are supported as `.npz` files. Semantic pixel-wise labels should always be provided in the same format as the data itself.

The directory structure in `$DATAPATH` is already standardized for both datasets (hence no rework is required for any new datasets, though of course it is quite likely that the dataset would have to be formatted accordingly.)

In general we expect `$DATAPATH` to be populated as follows:

```
<DATASET1>/
           raw/
               specific_original_data_format_1/...
            proc/
                 data/
                      train/
                            -  file_name_1.png
                            -  file_name_2.png
                      val/
                          - ...
                      test/
                          - ...
                 labels/
                      train/
                            -  file_name_1.png
                            -  file_name_2.png
                      val/
                          -...
                      test/
<DATASET2>/
           raw/
               specific_original_data_format_2/...
                  
```
Note that we expect to have data and labels the same file name. It's the parent directories `data/` and `labels/` which allow to differentiate between them.


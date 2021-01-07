# A User-Friendly Encoder-Decoder Meta-Search-Space For Semantic Segmentation

This repo allows to easily specify and search encoder-decoder architectures and associated hyperparameters. It is based on PyTorch and Ray Tune, an the asysnchronous successive halving algorithm (ASHA) as a search strategy.

The idea is to stack blocks: downsampling, bottleneck, and upsampling blocks.
Each block consists of several convolutional layers. The framework allows to stack an arbitrary amount of blocks, and an arbitrary number of convolutional layers within a block. The only requirement is to have an equal number uf downsampling and upsampling blocks, at least one bottleneck block, and at least one layer within bottleneck blocks. Please also see the below section "Building Blocks". 

**The current key use case is 

ating building robust, searched baselines for semantic segmentation tasks.**
Current restrictions are no data augmentation mechanisms and no ResNet-like or DenseNet-like connections between convolutional layers.


## Quickstart

1. Clone this repository: ```https://github.com/jmsckv/EncDecMeta.git && cd EncDecMeta ```
This code is tested with CUDA 10.2, Python 3.7.7 and setuptools 20.3.3 on Ubuntu 18.04. Higher versions should generally be supported.

2. We recommend to launch a Docker container with `. build_and_run_docker.sh` (use `_cpu.sh` if no GPU is available).  This will automatically create the expected directory structure and environment variables. It also auto-detects free ports for JupyterLab ($PORT1), Tensorboard ($PORT2), and the Ray Dashboard ($PORT3). Run `docker ps` to see where to retrieve e.g. JupyterLab in your browser, the default password, which you can change in `jupyter_notebook_config.py` before launching the container, is ASHA2020.

3. Create a Python virtutal env to install the project libraries. Do so from $CODEPATH in the Docker container, which maps to the root of this repo.

```
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install encdecmeta
# pip install -e . # run this instead the previous command to install in editable mode
```

4. Specify $PROC_DATAPATH which should map to the preprocessed data. Below, in the section Data Layout, we describe in depth the naming conventions we expect. In the Docker container this env variable is automatically set. It maps $CODEPATH/data/proc within the container to EncDecMeta/data/proc on your local disk.

5. Specify $RESULTSPATH where any experimental results are being stored. In the Docker container this env variable is automatically set. It maps $CODEPATH/results within the container to EncDecMeta/results on your local disk.

6. Run Experiments with `$CODEPATH/src/sample_and_train.py <YOUR_CONFIG.py>.` <YOUR_CONFIG.py> must be a .py file containing a dictionary named config. You can look at the Python files in `$CODEPATH/src/configurations/` to learn about specifying a configuration dictionary.

7. During training you can monitor the progress with `pip install tensorboard && tensorboard --logdir $RESULTSPATH --bind_all`.

### Tips & Tricks

Especially to familiarize yourself with the framework, you may want to modify the config file by adding or combining the following options:
- `config['verbose']=True` to gain more insight on what is going in the background, e.g. metrics are being calculated
- `'overfit['overfit']=True` to overfit the model on 1 val=train sample, which allows to check that the gradient updates work correctly
- `'overfit['num_samples']=X` with e.g. X=5 to restrict the training to 5 training/val samples which is useful if you want to simulate the outcomes of a search

Further you can find out about configurable hyperparameters for which a default value is set by `cd $CODEPATH && grep -r config.get`.


## Example: Unet

We can define an architecture close to the U-net proposed by Ronneberger et al. (2015) as follows:

```
c = ('C',3)

config = {'experiment_name': 'unet_fixed',
'D_blocks': [[c],[c],[c],[c],[]], # downsampling blocks
'B_blocks': [[c]],  # bottleneck blocks
'U_blocks': [[c],[c],[c],[c], [c,c]], # upsampling blocks
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

Train this model with `$CODEPATH/src/sample_and_train.py $CODEPATH/src/configurations/unet.py`
See the .py file for a more detailed discussion on differences to the original Unet.

Overall there are 5 downsampling and upsampling blocks as well as one bottleneck block.

## Example: Tweaking UNet

We can easily manually tweak an existing architecture. For example, we may alter the following model by including three more layer types
- `c2 = ('C', 2)` #  a convolution with dilation rate 2, which we use in the lower blocks of the network
- `h3 = ('H', 3)` #  a 1x3 convolution with dilation rate 3, which we use in the lower blocks of the network to capture long-range dependencies
- `h3 = ('V', 3)` #  a 3x1 convolution with dilation rate 3, which we use in the lower blocks of the network

Also we may want to add more convolutional filters in the first blocks.
We could then reformulate the net for example as:

```
'D_blocks': [[c,c,c],[c,c,c],[c2,c2],[c,c2],[c2,c2]]
'B_blocks': [[h3,v3,h3,v3,h3,v3]]
'U_blocks': [[c,c2],[c2,c2],[c2,c,2],[c,c,c],[c,c,c]]
```


## Example: Search Unets

Instead of deciding for this fixed architecture, we can embed the above model in a search space (cf. `src/configurations/unet.py`) by altering the above dictionary as follows:

```
c = (['H','V','C','O'], range(1,8)) # sampled layer, discussed below

config['experiment_name'] = 'unet_searched'
config['num_samples'] = 500 # evaluating 500 samples from this search space
config['dropout_ratio']: (0,0.5) # continuous hyperparameter
config['momentum']: (0.5,1) 
config['momentum_bn']: (0,1)
config['lr']: [i*j for i in [1,3,5,7] for j in [0.1, 0.01, 0.001]], # discrete hyperparameter
config['weight_decay']: [i*j for i in [1,3,5,7] for j in [0.01, 0.001, 0.0001]]
config['nesterov']: [True,False]
config['base_channels']: range(32,65)  # sample from range > discrete hyperparameter
config['batch_size']: range(1,11)
```

We search jointly for a good configuration of the SGD optimizer, regularization, and architecture. Batch size and number of base channels can generally result in OOMs. In this case, simply another candidate will get sampled, no manual intervention is required.

In general, we sample uniformly at random and from either lists, range objects or tuples. (Note: naively, you could model other distributions through repeating elements in a list).

A tuple describes a convolutional layer if its first entry is a list containing a combination of {'H', 'V','C','O'}, the second one a list of integers or a range object.
In this case, we sample from both tuple entries independently: first a layer operation, second the dilation rate.

Currently, there are 4 operations supported: 
- `'H'` (horizontal) maps to a `1x3 convolution`
- `'V'` (vertical) maps to a `3x1 convolution`
- `'C'` to a `3x3 convolution`
- `'O'`(one-by-one) to a `1x1 convolution`

So by adjusting `c = (['H','V','C','O'], range(1,8))`, we now describe a layer with 3 * 7 + 1 = 22 architectural decisions (for 'O' we ignore the sampled dilation rate).
W.r.t to the above Unet, we hence now describe a search space of 22**11 = 5.843183e+14 discrete architectures, the other sampled hyperparameters not counted.

Note that we could easily fix parts of an encoder-decoder network while searching others, e.g. only search the last upsampling block.


## Building Blocks

The search space consists of three abstractions:

- `Downsampling Blocks` halving the resolution of incoming feature maps while doubling the number of channels. The first operation in such a block is always hard-coded to be a 3x3 convolution with stride 2. After this layer, an arbitrary number of layers can be specified within the block.

- `Bottleneck Blocks` keeping both the number of feature maps and the spatial resolution constant. Within each of these blocks at least on layer must be specified.

- `Upsampling Blocks` always double the spatial resolution while halving the number of outgoing feature maps compared to the previous block. In these blocks the first two layers are hardcoded. Firstly, a 1x1 depthwise convolution fuses feature maps from the previous decoder block and the horizontally skip-connected encoder blocks of the same spatial resolution. Then a 3x3 transpose convolution with stride 2 guarantees the upsampling by a factor of two. Afterwards, an arbitrary number of layers can be horizontally stacked.


## Data - How to format your datasets.

The framework expects RGB images are supported  as .png files. Also the labels must be be formatted as such, but with only one channel.
Note that we expect to have data and labels the same file name. It's the parent directories data/ and labels/ which allow to differentiate between them.

In general we suggest $DATAPATH to be populated as follows and require $PROC_DATAPATH to map to /proc:

```
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

                  
```


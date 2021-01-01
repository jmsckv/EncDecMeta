# Defining Architectures and Search Spaces

This is a guide on how to define experiments such that they can be executed by `$PYTHONPATH/sample_and_train.py`.
All experiments have to be defined inside a `config` dictionary within a `.py` file, located in this folder.

Architecture and hyperparameter search space are defined analogously to training fixed architecures. 
In fact, you can think of the latter as a special case of a search, with a single possible choice for every possibly sampled entry in `config`.

Most importantly, in order to train fixed architectures, a flag 
`config['fixed_architectures']=True` must be set. When searching, this flag must be set to False.
This flag leads to training one architecture for max epochs on a given dataset (all defined in the configuration file).
If not set or set to False, Asychronuous Successive Halving as defined in the nested dictionary `configuration['tune']` (and possibly modified by the optional command line arguments of  `$PYTHONPATH/sample_and_train.py`) gets executed.


# Fixed Architectures

- `chargrid.py` is the implementation of the architecture presented in the Chargrid paper. It contains more than 40k training samples. The other experiments inherit from this configuration (i.e. the configuration is imported by these samples) and then overwrite it as applicable. If you train with large batch size on a P100 GPU, you adjust this by uncommenting the corresponding settings.

- `chargrid_10k.py`, is the same as the previous, but with a reduced training set of 10k samples. It serves as an ablation study for `search_arch_and_HPs.py` (see below) and tells us what performance gap to expect if searching for a Chargrid architecture on a reduced dataset.

- `ablation_HVH.py` is an ablation study w.r.t. to the original Chargrid architecture, where we replace all consecutive 3\*3 convolutional layers that are not responsible for down- or upsampling by a sequence of 1\*3  and 3\*1 convolutions. The character H ("horizontal") encodes a 1\*3 convolution, V ("vertical") a 3\*1 convolution. In the downsampling and upsampling blocks, we replace only two layers with a "HV" sequence. In the middle two encoder blocks of Chargrid, we replace all three layers with "HVH".

- `ablation_VHV.py` corresponds to the previous ablation study but with a different order of stacked asymmetric convolutions.
- `ablation_O.py`, replaces all non-up- or downsampling 3\*3 convolutions with one-by-one convolutions.

# Searched Architectures

There are two search spaces defined and examined in the scope of the thesis.

- `search_arch.py`covers a discrete architecture search space while all other, non-architectural hyperparameters are kept constant. It is constructed in such a way that we explicitly search for architectures that have fewer or at most an equal number of parameters compared to the reference architecture.
Non-downsampling 3\*3 convolutions with a fixed dilation rate are being replaced by a hierarchical uniform random sampling of first drawing an operation out of asymmetric convolutions (1/*3 or 3\*1), standard 3\*3 convolutions, or 1\*1 convolutions.  And then, secondly, sampling a dilation rate of up to 8 (does not apply to 1\*1 convolutions). In the case of upsampling layers, we randomly choose either a transposed 3\*3 convolution as in the reference architecture or a bilinear upsampling layer.

- `search_arch_and_HPs.py` covers a slightly smaller architectural search space excluding 1\*1 convolutions. 
However, also other non-architectural hyperparameters are now being included, e.g. the number of base channels, the batch size, or dropout rate. This may result in architectures with a larger memory footprint. Out-of-memory errors by sampling to large architectures are covered by a mechanism inside Ray Tune.

  
 # Getting Started
 
 Please have a look at `chargrid.py`, which contains the base configurations. 
 Do not modify this file. Instead, you can experiment by modifying the config dictionary imported in `my_experiment.py`.

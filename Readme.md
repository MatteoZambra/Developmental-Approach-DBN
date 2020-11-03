# Tutorial

This code allows to train a Deep Belief Network with the dimensions specified by the user and according to the dataset. The whole work has been thought to support the MNIST dataset and the SZ dataset, available on [OSF](https://osf.io/h5pfm/) website. For the SZ dataset specifications, please refer to Testolin A, Zou WY, McClelland JL.
Numerosity discrimination in deep neural networks: Initial competence, developmental refinement and experience statistics. Dev Sci. 2020;00:e12940. [DOI: https://doi.org/10.1111/
desc.12940](https://doi.org/10.1111/desc.12940)

## Data
The user only needs to set the proper flags in the main script. Instructions therein.

The user should download the SZ dataset (`.mat` files) and save it in the `dataset` directory. Then the program flow will manage to transform these files (and the MNIST data) in a `torch.Tensor` data structure, and save it in the same directory.

### MNIST Dataset

If the `torch.Tensor` dataset, saved as `pickle` serialzed file, is not present in the `dataset` directory, then set the variable `data_specs['stream']` to `'fetch'` and `data_specs['save']` to `True`. In the next program run, `data_specs['stream']` can be set to `'fetch'`, so to save some computational time, otherwise required to create and format properly the dataset tensor.

### SZ dataset

Here it sufficies to put the `.mat` files `SZ_data.mat` and `SZ_data_test.mat` in the `dataset` directory. Then `data_specs['streams']` can be left to be `'fetch'` (the data are already present, only matter of reformatting them).

## Train

Set `global_specs['train']` to `True` to train the model.

Different possibilities: use both the initialization schemes (normal, Glorot), and both the learning algorithms (greedy, iterative): lines 233 and 251 respectively. Insert the desired labels in the list to loop on. In the dictionary of gloabal variables `train_specs` it is possible to set the hyperparameters values, verbosity, dropout. 

## Global variables

The dictionary `global_specs` is the most important control structure of the program. 

There are many tasks that the program can perform:
  * Train the DBN(s);
  * Obtain the readout accuracy at the top of each layer in the DBN;
  * Perform the psychometric analyses (numerosity discrimination or estimation);

Note that in one run one can train the DBN, obtain the readout and perform the psychometric analyses. It may be of interest to have the psychometric analyses **during** the training stage, meaning that the user can set a list of reference epochs in which performs these analyses.

## Psychometric analyses

Can be numerosity estimation or discrimination. Depending on that, the linear classifier chosen for each task is different: perceptron with one single layer for estimation and delta rule for the discrimination. The hyperparameters of these classifiers can be set at the `if` statement in line 155-176.

## Plots

The structure `plot_specs` contains the global variables of interest.

The last segment of the program run plots the trends of MSE, readout, receptive fields, reconstructions of the data samples to assess the DBN performance. 

______

##### Note. 
The steps one would do are:
  1. Train the DBNs, say 10 runs, so to be able to average the performance metrics trajectories. In this way, the DBN performance can be tracked;
  2. Once the good behavior of the DBN training is assessed, then one run for the psychometric analyses sufficies. Set the epochs during training, so to assess the development of number acuity sense;
  3. Plots;
  4. If needed, network science analyses.

______

## Trubleshooting

Some recurring errors may pop up.

### Path
Errors concerning paths may occur, eg. `Errno4: File not found`. In this case make sure that the model has been saved, and the folders are properly named.

### Serialization
It is warmly suggested to use an hardware acceleration. Pytorch can manage it, but be careful that a model saved as GPU (meaning that the parameters are `torch.Tensor` created to be handled by the GPU), then can NOT be fetched on a only-CPU machine.

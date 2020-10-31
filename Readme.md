# Tutorial

This code allows to train a Deep Belief Network with the dimensions specified by the user and according to the dataset. The whole work has been thought to support the MNIST dataset and the SZ dataset, available on [CCNL](http://ccnl.psy.unipd.it/research/deeplearning) website. 

## Data
The user only needs to set the proper flags in the main script. Instructions therein.

The user should download the SZ dataset (`.mat` files) and save it in the `dataset/` directory. Then the program flow will manage to transform these files (and the MNIST data) in a `torch.Tensor` data structure, and save it in the same directory.

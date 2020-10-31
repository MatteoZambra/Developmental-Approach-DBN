
import os
import sys
sys.path.append(os.getcwd() + r'\dbp')

import pandas as pd
import pickle
import dbns
import gcdata
import gcvis
import utls
import torch
import time
import datetime


# GPU -------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#end
print('Using {}'.format(device))

# DATA ------------------------------------------------------------------------
# among 'mnist' and 'sz'
dataset_id = 'mnist'

path_data = os.getcwd() + r'/dataset'
data_specs = {'device'     : device,

              # NOTE that optimal batch size depends much on the hardware
              # these values have been observed to work well
              'batch_size' : 160 if dataset_id == 'sz' else 100,
              
              # MNIST: set `stream' to `fetch' if the dataset is available yet. Set to `create' if not
              # SZ: set to `fetch' in any case. The .mat files are fetched and processed
              'stream'     : 'fetch',
              
              # MNIST: If streams == fetch, then set `save' to `True' to save the torch.Tensor copy of the dataset
              'save'       : False,
              
              # MNIST: if `binarize' is set to 'p', then the visible activities are NOT binarized,
              # otherwise the activities are binarized according to `factor'
              # Eg if factor is set to 3, then the activities X_ij < 1/3. are set to 0, the others are set to 1
              'binarize'   : 'p',
              'factor'     : 3.,
              'path'       : path_data}

if dataset_id == 'mnist': dataset = gcdata.MNISTdata(data_specs)
if dataset_id == 'sz':    dataset = gcdata.SZdata(data_specs)


# SETUP -----------------------------------------------------------------------
if dataset_id == 'mnist': layers = [dataset.GetNfeats(), 500, 500, 2000]
if dataset_id == 'sz':    layers = [dataset.GetNfeats(), 80, 400]

# Hyperparameters, feel free to fiddle, 
train_specs = {'epochs'           : 100 if dataset_id == 'sz' else 50,
               'initial_lr'       : 0.1 if dataset_id == 'sz' else 0.01,
               'final_lr'         : 0.1 if dataset_id == 'sz' else 0.01,
               'weight_decay'     : 0.0002 if dataset_id == 'sz' else 0.0001,
               'initial_momentum' : 0.5,
               'final_momentum'   : 0.9,
               
               # dropout: p \in [0,1]
               'dropout'          : 1,
               
               # Verbosity: if yes, then the accuracy, loss and details are printed for each epoch.
               # If you have to train for different runs, then it is better set verbose to False,
               # so that you see at which point training is.
               # NOTE: maybe try to train a model with one run before, to check that training is 
               # done correctly, then set off the verbosity
               'verbose'          : True,
               
               # Progressive train: DEPRECATED
               'prog_train'       : False,
               
               # If binarize activities is set to true, then during the MCMC sampling, the 
               # visible activities are binarized, accoring to a Bernoulli sampling
               'binarize_act'     : False}

global_specs = {
                # General arguments 
                
                # Train: Execute the training stage of the model, in which the 
                # numerosity estimation or discrimination is included -- NumDiscr/Estim is done 
                # in a training-wise fashion: during the training of the model, the discrimination
                # or estimation performance is assessed
                'train'         : False,
                
                # Post process: performance (loss, readout accuracy metrics) plots, plot of the
                # receptive fields, of the recostructed samples
                'analysis'      : True,
                
                # NOTE: READOUT IS SLOW! Because is coded with scikit-learn API, which does not 
                # take advantage of hardware acceleration. 
                # IF YOU DO NOT NEED READOUT, then SET THIS FLAG TO FALSE, for goodness! 
                # The model will train much faster
                'readout_acc'   : False,
                'verbose'       : train_specs['verbose'],

                # Training (NOT model) meta-parameters
                'runs'          : 10,
                'mcmc_schedule' : [1],
                
                # Whether to save training-in-progress DBN
                # This is done for the network science part. 
                'save_tmp'      : False,
                'dropout'       : train_specs['dropout'],
                
                # Glorot correction: down-scales the weight matrices in case of Glorot intialization,
                # if needed
                'gcorrection'   : 0.1,
                'epochs'        : train_specs['epochs'],
                
                # If save_tmp is set to True, then these epochs are those in which the not-fully-trained-model is saved
                # NOTE: While greedy training involves one layer at time, iterative updates all the parameters in one epoch.
                # In order to plot the network variables (average degree, average geodesic distance) with respect to ``time'',
                # it is necessary to suitably choose the epochs. 
                'epochs_iter'   : range(train_specs['epochs']) if dataset_id == 'minst' else [0, 5, 10, 20, 30, 50, 75, 90],
                'epochs_glws'   : [5, 10, 20, 30, 40] if dataset_id == 'minst' else [0, 5, 10, 20, 30, 50, 75, 90],
                'dataset'       : dataset_id,

                # Linear classifier and Weber fraction estimation instructions
                # Instruct the program whether do the discrimination or estimation task.
                # numerosity_last_epoch if set to true performs the analysis only on the
                # last epoch, when model is fully trained.
                # Otherwise, the developmental-wise analyses are done. 
                # The reference epochs are specified below
                'numerosity_discrimination'   : False,
                'numerosity_estimation'       : True,
                'numerosity_last_epoch'       : False,
                # 'web_nref'                  : 16, # DEPRECATED
                'progressive_lc_fit'          : train_specs['prog_train'],
                
                # Dictionary having as keys the reference epochs in which the ``psychometric''
                # analyses are done. The items are the results.
                # For estimation, we have the Weber fractions referred to each run and each realization
                # of the linear classifier. In fact, the linear classifiers trained to do this are a number
                # specifier by the user, so to average the results in the end
                'psydata_df'                  : dict(),
                
                # number of linear classifiers. 5 is recommended
                'num_linear_classifiers'      : 1,
                
                # The number of epochs to train the linear classifier with
                'lc_train_epochs'             : 500,
                
                # General purposes paths
                'path_weber'    : os.getcwd() + r'/weberfrac',
                'path_images'   : os.getcwd() + r'/images'}


if global_specs['numerosity_estimation']:
    
    # Here: in the case of estimation, the linear classifier should produce an answer +
    # that is the label of the input data
    # To initialize the linear classifier, the global variable 'data_classes' needs to know
    # how many output classes we have in the dataset, so to create a single layer perceptron 
    # with many output units as the data classes
    global_specs['data_classes'] = dataset.getLabelsRange().__len__()
    
    # in fact, the MLP (one layer) is the classifier of choice in this case
    global_specs['classifier'] = 'MLP'
    
    # hyperparameters of the linear classifier
    global_specs['lc_hyperparams'] = {'learning_rate'    : 0.01 if dataset_id == 'mnist' else 0.1,
                                      'weights_decay'    : 0.00001 if dataset_id == 'mnist' else 0.00000001,
                                      'momentum_initial' : 0.4,
                                      'momentum_final'   : 0.85 }
    
    # And here we have the ``names'' of the classes
    global_specs['estimation_range'] = dataset.getLabelsRange()
    
elif global_specs['numerosity_discrimination']:

    # In the case of discrimination (the number of points in the figure is smaller or greater than reference)
    # the classifier only has one output unit. 
    global_specs['data_classes'] = 1
    
    # and the classifier si simply a delta rule
    global_specs['classifier'] = 'DeltaRule'
    
    # Here: the users sets the (ONLY TWO *) number of reference, and the associated
    # ranges.
    #---
    # * In the file utls.py, in the function plotPsychometricCurves, the collection
    # of psychometric data is hardcoded (sorry) so to have only two number of reference.
    # If the user wished, he could modify the code so to have more 
    global_specs['discrimination_Nref_ranges'] = {8 : range(5,13), 16 : range(10,25)}
    global_specs['lc_hyperparams'] = {'learning_rate'    : 0.000001,
                                      'weights_decay'    : 0.000001,
                                      'momentum_initial' : 0.5,
                                      'momentum_final'   : 0.9 }
#end

# The epochs at which perform the numerosity estimation/discrimination tasks
epochs_lc = [0, 5, 10, 20, 30, 50, 70, 99]
if global_specs['numerosity_last_epoch']:
    global_specs['lc_epochs_stamps'] = [train_specs['epochs'] - 1]
else:
    global_specs['lc_epochs_stamps'] = epochs_lc
#end

# DEPRECATED
global_specs['metrics'] = {
                      8  : pd.DataFrame(index = global_specs['lc_epochs_stamps'], columns = ['TrainAcc', 'TrainLoss']),
                      16 : pd.DataFrame(index = global_specs['lc_epochs_stamps'], columns = ['TrainAcc', 'TrainLoss'])}

# In the phase of Analyses, the plot specifications are required
plot_specs = {'path_images'   : os.getcwd() + r'/images/',
              'path_models'   : os.getcwd() + r'/models/',
              'save_pictures' : True,
              'plot_digits'   : True,
              'runs'          : global_specs['runs'],
              'gcorrection'   : global_specs['gcorrection'],
              'layers'        : len(layers) - 1,
              'epochs'        : train_specs['epochs'],
              'dropout'       : train_specs['dropout'],
              'dataset_id'    : dataset_id}


# TRAIN -----------------------------------------------------------------------
# either training both or one, here starts training, hence here the start time
# stamp is placed


if global_specs['train']:
    
    start_ = time.time()
    
    for scheme in ['normal']:
        
        global_specs['scheme'] = scheme
        print('Init scheme : {}'.format(scheme))
        
        path = os.getcwd() + r'/models/series_{}runs_{}'.format(global_specs['runs'], scheme)
        global_specs['pathseries'] = path
        
        for steps in global_specs['mcmc_schedule']:
            
            train_specs['mcmc_steps'] = steps
            
            for run in range(global_specs['runs']):
                
                global_specs['nrun'] = run
                global_specs['psydata_df'].update({run : pd.DataFrame(columns = range(global_specs['num_linear_classifiers']), index = global_specs['lc_epochs_stamps'])})
                print('\n---\nRUN : {}'.format(run))
                
                for alg in ['iterative']:
                    
                    global_specs['algorithm'] = alg
                    global_specs['pathtmp'] = os.getcwd() + r'/models/epochs_tmp/{}/'.format(scheme)
                    
                    dbn = dbns.DeepBeliefNetwork(layers, global_specs)
                    dbn.setRunID(run)
                    dbn.train(dataset, train_specs, global_specs)
                    
                    # NOTE: If you do not need the program to save the models, just comment the line below
                    dbn.save_model(global_specs, epoch = train_specs['epochs'])
                    
                # end aa loop
            #end rr loop
        #end end st loop
    #end sc loop
    
    
    finish_ = time.time()
    print('\nTotal training time: {} hh:mm:ss'.format(datetime.timedelta(seconds = finish_ - start_)))
    
    if global_specs['numerosity_discrimination'] or global_specs['numerosity_estimation']:
        with open(os.getcwd() + r'/saveddetails/fresh/wdf.pkl', 'wb') as f:
            pickle.dump(global_specs['psydata_df'], f)
        #end
        f.close()
    #end
#end


# ANALYSES --------------------------------------------------------------------
# If you only need to see the performance metrics, uncomment the for scheme loops,
# both for MSE averages and receptive fields
if global_specs['analysis']:
    plot_specs['mcmc_steps'] = 1
    
    gcvis.allProfiles(['normal', 'glorot'], ['greedy', 'iterative'], plot_specs, metrics = ['readout', 'cost'])
    
    for scheme in ['normal', 'glorot']:
      plot_specs['scheme'] = scheme
      print('Scheme: {}'.format(scheme))
      for loop in [1]:
          plot_specs['loops'] = loop
          utls.MSE_averages(dataset.GetXtest(), dataset.GetYtest(), plot_specs)
      #end
    
    for scheme in ['normal']:
        plot_specs['scheme'] = scheme
        utls.receptive_fields(plot_specs)
    #end
#end







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
dataset_id = 'mnist'  # among 'mnist' and 'sz'

path_data = os.getcwd() + r'/dataset'
data_specs = {'device'     : device,
              'batch_size' : 160 if dataset_id == 'sz' else 100,
              'stream'     : 'fetch',
              'save'       : False,
              'binarize'   : 'p',
              'factor'     : 3.,
              'path'       : path_data}

if dataset_id == 'mnist': dataset = gcdata.MNISTdata(data_specs)
if dataset_id == 'sz':    dataset = gcdata.SZdata(data_specs)


# SETUP -----------------------------------------------------------------------
if dataset_id == 'mnist': layers = [dataset.GetNfeats(), 500, 500, 2000]
if dataset_id == 'sz':    layers = [dataset.GetNfeats(), 80, 400]

train_specs = {'epochs'           : 100 if dataset_id == 'sz' else 50,
#                'initial_lr'       : 0.1 if dataset_id == 'sz' else 0.01,
#                'final_lr'         : 0.1 if dataset_id == 'sz' else 0.01,
#                'weight_decay'     : 0.0002 if dataset_id == 'sz' else 0.0001,
               # 'initial_momentum' : 0.5,
               # 'final_momentum'   : 0.9,
               'initial_lr'       : 0.001,
               'final_lr'         : 0.001,
               'weight_decay'     : 0.0000001,
               'initial_momentum' : 0.4,
               'final_momentum'   : 0.85,
               'dropout'          : 1,
               'verbose'          : True,
               'prog_train'       : False,
               'binarize_act'     : False}

global_specs = {
                # General arguments 
                'train'         : False,
                'analysis'      : True,
                'readout_acc'   : False,
                'verbose'       : train_specs['verbose'],

                # Training (NOT model) meta-parameters
                'runs'          : 10,
                'mcmc_schedule' : [1],
                'save_tmp'      : False,   # Whether to save training-in-progress DBN
                'dropout'       : train_specs['dropout'],
                'gcorrection'   : 0.1,
                'epochs'        : train_specs['epochs'],
                'epochs_iter'   : range(train_specs['epochs']) if dataset_id == 'minst' else [0, 5, 10, 20, 30, 50, 75, 90],
                'epochs_glws'   : [5, 10, 20, 30, 40] if dataset_id == 'minst' else [0, 5, 10, 20, 30, 50, 75, 90],
                'dataset'       : dataset_id,

                # Linear classifier and Weber fraction estimation instructions
                'numerosity_discrimination'   : False,
                'numerosity_estimation'       : True,
                'numerosity_last_epoch'       : False,
                # 'web_nref'                  : 16,
                'progressive_lc_fit'          : train_specs['prog_train'],
                'psydata_df'                  : dict(),
                'num_linear_classifiers'      : 1,
                'lc_train_epochs'             : 500,
                
                # General purposes paths
                'path_weber'    : os.getcwd() + r'/weberfrac',
                'path_images'   : os.getcwd() + r'/images'}


if global_specs['numerosity_estimation']:
    global_specs['data_classes'] = dataset.getLabelsRange().__len__()
    global_specs['classifier'] = 'MLP'
    global_specs['lc_hyperparams'] = {'learning_rate'    : 0.01 if dataset_id == 'mnist' else 0.1,
                                      'weights_decay'    : 0.00001 if dataset_id == 'mnist' else 0.00000001,
                                      'momentum_initial' : 0.4,
                                      'momentum_final'   : 0.85 }
    global_specs['estimation_range'] = dataset.getLabelsRange()
    
elif global_specs['numerosity_discrimination']:
    global_specs['data_classes'] = 1
    global_specs['classifier'] = 'DeltaRule'
    global_specs['discrimination_Nref_ranges'] = {8 : range(5,13), 16 : range(10,25)}
    global_specs['lc_hyperparams'] = {'learning_rate'    : 0.000001,
                                      'weights_decay'    : 0.000001,
                                      'momentum_initial' : 0.5,
                                      'momentum_final'   : 0.9 }
#end

# epochs_lc = [0, 1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 35, 40 ,45, 50, 75, 99]
epochs_lc = [0, 5, 10, 20, 30, 50, 70, 99]
if global_specs['numerosity_last_epoch']:
    global_specs['lc_epochs_stamps'] = [train_specs['epochs'] - 1]
else:
    global_specs['lc_epochs_stamps'] = epochs_lc
#end

global_specs['metrics'] = {
                      8  : pd.DataFrame(index = global_specs['lc_epochs_stamps'], columns = ['TrainAcc', 'TrainLoss']),
                      16 : pd.DataFrame(index = global_specs['lc_epochs_stamps'], columns = ['TrainAcc', 'TrainLoss'])}


plot_specs = {'path_images'   : os.getcwd() + r'/images/',
              'path_models'   : os.getcwd() + r'/models/',
              'save_pictures' : True,
              'plot_digits'   : True,
              'runs'          : global_specs['runs'],
              'gcorrection'   : global_specs['gcorrection'],
              'layers'        : len(layers) - 1,
              'epochs'        : train_specs['epochs'],
              'dropout'       : train_specs['dropout'],
              'dataset'       : dataset_id}


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
                    # dbn.save_model(global_specs, epoch = train_specs['epochs'])
                    
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
if global_specs['analysis']:
    plot_specs['mcmc_steps'] = 1
    
    # gcvis.allProfiles(['normal', 'glorot'], ['greedy', 'iterative'], plot_specs, metrics = ['readout', 'cost'])
    
    # for scheme in ['normal', 'glorot']:
    #   plot_specs['scheme'] = scheme
    #   print('Scheme: {}'.format(scheme))
    #   for loop in [1]:
    #       plot_specs['loops'] = loop
    #       utls.MSE_averages(dataset.GetXtest(), dataset.GetYtest(), plot_specs)
    #   #end
    
    for scheme in ['normal']:
        plot_specs['scheme'] = scheme
        utls.receptive_fields(plot_specs)
    #end
#end







'''
Class Deep Belief Network
-------------------------

Containts methods for training a DBN with the classic greedy layer-wise 
algorithm and with the novel joint iterative training. 
Either way, the metric used to optimize the parameters is Contrastive Divergence.
It is thought to suffice for now to rely on such a well documented an proven way.

Contains specific informations regarding:
    * numerical values of the layers of neurons and the 
      respective RBM models
    * global variables such as the initialization scheme, algorithm...
    
NOTE: The trianing method is the Achille's heel of the system. It would be
more compact and elegant to write an optimizer class, but the substantial
difference between the greedy and iterative algorithms makes it far from 
obvious. For ease of coding, in this specific work, a training method has 
been written for the greedy case (which relies heavily on the RBM class) and
for the iterative counterpart, re-using the RBM training based on CD.
'''

import torch
from torch.nn import MSELoss
import time

import numpy as np
import rbms
import random
import utls
import datetime


class DeepBeliefNetwork:
    
    def __init__(self, layers, global_specs):
        '''
        Initialization of the DBNet.
        Input:
            - layers (list): contains the numerical values of the number of 
                             units involved in each layer
            - global_specs (dict): global variables and specifications. E.g. the
                                   initialization scheme, algotithm, dropout fraction, ...
        Output:
            none
        '''
        
        self.layers     = layers
        self.nlayers    = len(layers) - 1
        self.rbm_layers = list()
        self.algorithm  = global_specs['algorithm']
        self.scheme     = global_specs['scheme']
        self.train_time = 0.0
        self.run_id     = 0
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i in range(1, self.nlayers + 1):
            
            layer_id = i - 1
            input_size  = layers[layer_id]
            output_size = layers[i]
            self.rbm_layers.append(rbms.RestrictedBoltzmannMachine(input_size, output_size, layer_id, global_specs))
            if i == self.nlayers:
                self.rbm_layers[layer_id].last_layer = True
            #end
        #end
    #end
    
    def setRunID(self, run):
        '''
        Set the run identificative number. 
        Input:
            - run (int): so to uniquely save the model in case of multiple runs
        Output:
            none
        '''
        
        self.run_id = run
    #end
    
    def forward(self, v):
        '''
        Propagate the input batches up to the last layer
        Input:
            - v (torch.Tensor): input vectors (in batches)
        Output:
            - _v (torch.Tensor): activities of the last layer (probabilities)
        '''
        
        _v = v.clone()
        for rbm in self.rbm_layers:
            _v, _ = rbm.forward(_v)
        #end
        return _v
    #end
    
    def train(self, dataset, train_specs, global_specs):
        '''
        Model training. Here the cases accounted for are greedy and iterative learning.
        If training is greedy, then the main loop is on the RBM layers, and each layer
        is trained as it were a simple RBM and the dataset is propagated to the next layer.
        Else if training is iterative, then the main loop is on the epochs, and the signal 
        travels all along the model, the layers of which are updated all at once. Meaning
        that once one layer is touched, the parameters are updated, but only once. It is 
        during the next epoch that the same parameters will be modified once more.
        Important to note that the RBMs are initialized when the DBN is initialized. After, 
        they are gradually updated.
        Input:
            - dataset (gcdata.DataLoader child, either MNISTdata or SZdata): dataset
            - train_specs (dict): training specifications. Learning rate, ...
            - global_specs (dict): global variables
        Output:
            - none
        '''
        
        self.train_specs = train_specs
        Xtrain = dataset.GetXtrain()
        Xtest  = dataset.GetXtest()
        Ytrain = dataset.GetYtrain()
        Ytest  = dataset.GetYtest()
        
        if self.algorithm == 'greedy':
            
            print('\nTrain : {}\n'.format(self.algorithm))
            start = time.time()
            for rbm in self.rbm_layers:
                
                # in this case, the RBM is trained ordinarily.
                # the dataset is projected to the next layer and
                # used as dataset for the next RBM layer
                _Xtrain, _Xtest = rbm.train(self, Xtrain, Ytrain, Xtest, Ytest, train_specs, global_specs)
                Xtrain = _Xtrain.clone()
                Xtest  = _Xtest.clone()
            #end layers loop
            
            end = time.time()
            self.train_time = end - start
            print('\n--{} DBN. Total training time: {} hh:mm:ss'.format(self.algorithm, datetime.timedelta(seconds = end - start)))
            
        elif self.algorithm == 'iterative':
            
            print('\nTrain : {}\n'.format(self.algorithm))
            start = time.time()
            
            for rbm in self.rbm_layers: # INITIALIZE GRADIENTS
                rbm.dW = torch.zeros_like(rbm.W)
                rbm.da = torch.zeros_like(rbm.a)
                rbm.db = torch.zeros_like(rbm.b)
            #end
            
            for epoch in range(train_specs['epochs']):
                
                global_specs['current_epoch'] = epoch
                if train_specs['verbose']: print('\nEpoch {}'.format(epoch))
                self.iterativeTrain(Xtrain, Ytrain, Xtest, Ytest, epoch, train_specs, global_specs)
                
                if global_specs['save_tmp']:
                    if epoch in global_specs['epochs_iter']:
                        # if we need to save the partially learned model
                        self.save_model(global_specs['pathtmp'], epoch, global_specs)
                    #end
                #end
            #end epochs loop
            
            end = time.time()
            self.train_time = end - start
            print('\n--{} DBN. Total training time: {} hh:mm:ss'.format(self.algorithm, datetime.timedelta(seconds = end - start)))
        #end
    #end
    
    def iterativeTrain(self, Xtrain, Ytrain, Xtest, Ytest, epoch, train_specs, global_specs):
        '''
        Iterative joint training. 
        For each epoch, the RBM layers are visited and updated. 
        Input:
            - Xtrain, ... (torch.Tensor): dataset
            - epoch (int): epoch time stamp
            - train_specs, global_specs (dict): see above
        Output: none
        '''
        
        for rbm in self.rbm_layers:
            if train_specs['verbose']: 
                print('\nLayer {}: Input size, output size = ({},{})'.format(rbm.layer_id, rbm.Nv, rbm.Nh))            
            #end
            
            # At each epoch the mask vector for dropout is generated
            rbm.dromask = torch.Tensor(np.random.binomial(1, p = train_specs['dropout'], size = (Xtrain.shape[1], rbm.Nh)), device = self.device)
            
            _Xtrain = torch.zeros( [Xtrain.shape[0], Xtrain.shape[1], rbm.Nh] )
            _Xtest  = torch.zeros( [Xtest.shape[0], Xtest.shape[1], rbm.Nh] )
            
            # the gradients are zeroed for each layer --- BUT MAYBE NO
            # rbm.dW = torch.zeros_like(rbm.W)
            # rbm.da = torch.zeros_like(rbm.a)
            # rbm.db = torch.zeros_like(rbm.b)
            
            criterion = MSELoss(reduction = 'mean')
            train_loss = 0.0
            
            # project the test set BEFORE parameters update!
            for n in range(Xtest.shape[0]):
                _Xtest[n,:,:], _ = rbm.forward(Xtest[n,:,:])
            #end
            
            # batches shuffle
            batch_indices = list(range(Xtrain.shape[0]))
            random.shuffle(batch_indices)
            for n in batch_indices:
                
                # project the data set BEFORE parameters update!!
                _Xtrain[n,:,:] = rbm.forward(Xtrain[n,:,:].clone())[0].clone()
                pos_v = Xtrain[n,:,:].clone()
                loss = rbm.contrastive_divergence_params_update(criterion, pos_v, epoch, Xtrain.shape[1], train_specs)
                
                train_loss += loss
                
                # if train_specs['prog_train']:
                #     # if we are interested in monitoring the progressive learning 
                #     # of the DeltaRule classifier
                #     dr_loss, dr_acc = rbm.dr.fit(rbm.forward(pos_v)[0].clone(), Ytrain[n,:,:].clone())
                #     rbm.loss_finer.append(dr_loss)
                #     rbm.acc_finer.append(dr_acc)
                # #end
                
            #end batches loop
            
            rbm.cost_profile.append(train_loss / Xtrain.shape[0])
            if global_specs['readout_acc']: 
                rbm.readout_profile.append(rbm.linear_readout(_Xtrain, Ytrain, _Xtest, Ytest))
            #end
            
            if train_specs['verbose']: 
                print('Training loss = {:.6f}'.format(train_loss / Xtrain.shape[0]))
                if global_specs['readout_acc']: 
                    print('[RidgeClassifier] (Test)  Read-out accuracy at layer {:d}: {:.4f}'.format(rbm.layer_id, rbm.readout_profile[-1]))
                #end
            #end
            
            Xtrain = _Xtrain.clone()
            Xtest  = _Xtest.clone()
            
            # if epoch in global_specs['lc_epochs_stamps'] and rbm.last_layer:
                
            
            if global_specs['numerosity_discrimination'] and rbm.last_layer:
                # if we need to estimate the Weber fraction at intermediate stages of learning
                if epoch in global_specs['lc_epochs_stamps']:
                    for classifier in range(global_specs['num_linear_classifiers']):
                        print('\n\tClassifier {}'.format(classifier))
                        psydata = utls.plotPsychometricCurves(Xtrain, Ytrain, Xtest, Ytest, global_specs)
                        global_specs['psydata_df'][global_specs['nrun']].at[global_specs['current_epoch'], classifier] = psydata
                    #end
                #end
            elif global_specs['numerosity_estimation'] and rbm.last_layer:
                if epoch in global_specs['lc_epochs_stamps']:
                    for classifier in range(global_specs['num_linear_classifiers']):
                        print('\n\tClassifier {}'.format(classifier))
                        psydata = utls.getNumerosityEstimate(Xtrain, Ytrain, Xtest, Ytest, global_specs)
                        global_specs['psydata_df'][global_specs['nrun']].at[global_specs['current_epoch'], classifier] = psydata
                    #end
                #end
            #end
            
        #end layers loop
    #end
    
    
    def save_model(self, global_specs, epoch = '', rbml = ''):
        '''
        Save the serialized model on disk.
        Input:
            - global_specs (dict): which contains the path where we want to save the model
            - epoch (int): if this keyword is used, then the epoch time stamp is used to name
                           the saved model. Default is '', which has no effect on the name of
                           the model
            - rbml (int): same, but with the RBM layer identifier
        Output:
            none
        '''
        
        name = self.get_name(global_specs, epoch = epoch, rbml = rbml)
        
        for rbm in self.rbm_layers:
            del rbm.dW
            del rbm.da
            del rbm.db
        #end
        
        with open(global_specs['pathseries'] + name + '.pkl', 'wb') as handle:
            torch.save(self, handle)
            handle.close()
        #end
        
    #end
    
    def get_name(self, global_specs, epoch = '', rbml = ''):
        '''
        Get the name of the model, depending on the global variables, such as initialization scheme,
        algorithm, ...
        Input:
            - global_specs (dict): as above
            - epoch (int) and rbml (int): as above
        Output:
            - name (str): name of the model, used to save it
        '''
        
        droptag  = '_nodrop' if global_specs['dropout'] == 1 else '_drop{}'.format(str(global_specs['dropout']).replace('.','d'))
        gcorrtag = '_nogcorr' if global_specs['gcorrection'] == 1 else '_gcorr{}'.format(str(global_specs['gcorrection']).replace('.','d'))
        
        if rbml != '':
            if epoch != '':
                epoch = int(rbml) * global_specs['epochs'] + int(epoch)
                epoch = '_ep{}'.format(epoch)
            #end
        else:
            if epoch != '': epoch = '_ep{}'.format(epoch)
        #end
        
        name = r'/{}_{}_{}_dbn{}_cd{}{}{}{}'.format(global_specs['dataset'], global_specs['scheme'], global_specs['algorithm'],
                                               global_specs['nrun'], self.train_specs['mcmc_steps'], epoch, droptag, gcorrtag)
        
        return name
    #end

#endCLASS
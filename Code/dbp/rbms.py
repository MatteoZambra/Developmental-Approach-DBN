
'''
Class Restricted Boltzmann Machine
----------------------------------

Self-contained class to model the RBM neural network.
The idea followed is to encapsulate in this class the basic parameters
update rule, the Contrastive Divergence (Hinton, 2002). 

Contains specific informations regarding:
    * dimensions
    * the model parameters
    * loss and readout trends, referred to the hidden layer
      of the RBM. These are eventually framed in the scope of 
      the Deep Belief Network performance assessment
    * the Bernoulli sampled vector to implement dropout, if needed
    * DeltaRuleClassifier object, to implement the numerosity task
'''

import numpy as np
import torch
from torch import matmul, sigmoid, bernoulli
from torch.nn import MSELoss

import random
import linearclassifier as lc



class RestrictedBoltzmannMachine:
    
    def __init__(self, visible_dim, hidden_dim, layerid, global_specs):
        '''
        Initialization of the class instance.
        Input:
            - visible_dim, hidden_dim (int): model dimensions
            - layerid (int): layer identifier. It is used to save 
                             the model with a unique name
            - global_specs (dict): parameters initialization specifications,
                                   scheme and algorithm details, global variables
        Output:
            none
        '''
        
        self.Nh = hidden_dim
        self.Nv = visible_dim
        
        self.layer_id        = layerid
        self.last_layer      = False
        self.cost_profile    = list()
        self.readout_profile = list()
        self.acc_finer       = list()
        self.loss_finer      = list()
        self.params_init(global_specs['scheme'], global_specs['gcorrection'], global_specs['dataset'])
        
        self.dropmask = torch.ones(self.Nh)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # if global_specs['classifier'] == 'DeltaRule':
        #     self.linclass = lc.DeltaRuleclassifier(hidden_dim, global_specs['data_classes'], global_specs)
        # elif global_specs['classifier'] == 'MLP':
        #     self.linclass = lc.MLPclassifier(hidden_dim, global_specs['data_classes'], global_specs)
        # #end
    #end
    
    
    def params_init(self, scheme, gcorrection, dataset_id):
        '''
        Parameters initialization.
        Input:
            - scheme (str): specifier of the initialization scheme
            - gcorrection (float): amount of down-scaling for the Glorot weights
            - dataset_id (str): dataset identifier
        Output:
            none
        '''
        
        self.scheme = scheme
        
        if scheme == 'normal':
            
            if dataset_id == 'mnist':
                c = 0.01
            elif dataset_id == 'sz':
                c = 0.1
            self.W = torch.nn.init.normal_(torch.zeros(self.Nh, self.Nv), mean = 0, std = 1) * c
            
        elif scheme == 'glorot':
            
            self.W = torch.nn.init.xavier_normal_(torch.zeros(self.Nh, self.Nv)) * gcorrection
        #end
        
        self.a = torch.zeros(1, self.Nv)
        self.b = torch.zeros(1, self.Nh)
    #end
    
    
    def forward(self, v, dropfrac = 1):
        '''
        Implements the product W.v + b, with W the weight matrix, b the hidden layer bias.
        Input:
            - v (torch.Tensor): input vector -- Input to this RBM
            - dropfrac (float): probability of dropping a neuron. Default is 1, 
                                so that the needed value can be given when it 
                                is necessary
        Output:
            - p_h_given_v (torch.Tensor): vector of probabilities of the hidden layer
                                          The values of these probabilities is given by
                                              p(h|v) = sigm(v.W' + b)
                                          and the hidden activities can be sampled from 
                                          this distribution
            - h (torch.Tensor): hidden activities sampled from the former.
                                    h ~ p(h|v)
        '''
        
        p_h_given_v = sigmoid(matmul(v, self.W.t()) + self.b)
        h = bernoulli(p_h_given_v)
        
        if dropfrac < 1:
            p_h_given_v = p_h_given_v * self.dropmask
        
        return p_h_given_v, h
    #end
    
    
    def backward(self, h):
        '''
        Backward pass. Likewise, to obtain the probabilies and visible activities
        given the hidden layer activities.
        Input:
            - h (torch.Tensor): hidden activities to project back to the visible layer
        Output:
            - p_v_given_h (torch.Tensor): p(v|h) = sigm(h.W + a), the apex denotes the 
                                          transposition and a is the visible layer bias
            - v (torch.Tensor): v ~ p(v|h)
        '''
        
        p_v_given_h = sigmoid(matmul(h,self. W) + self.a)
        v = bernoulli(p_v_given_h)
        return p_v_given_h, v
    #end
    
    
    def Gibbs_sampling(self, v, mcmcsteps, dropfrac = 1):
        '''
        Block Gibbs Sampling. Performs alternate sampling between the two layers,
        so that the equilibrium distribution can be reached and the samples eventually
        obtained are close enough to samples from the equilibrium distribution.
        It has been shown that one steps gives satisfactory results.
        Input: 
            - v (torch.Tensor): visible pattern
            - mcmcsteps (int): number of sampling steps
            - dropfrac (float): dropout probability
        Output:
            - p_h (torch.Tensor): hidden probabilities
            - _v (torch.Tensor): sampled visible activities
            - p_v (torch.Tensor): visible probabilities
        
        NOTE: hidden activities binarized only in case of multiple 
        back and forth sampling steps!
        '''
        
        _v = v.clone()
        for k in range(mcmcsteps):
            
            p_h, h = self.forward(_v, dropfrac)
            if (k == mcmcsteps-1 and mcmcsteps > 1): 
                h = p_h
            p_v, _v = self.backward(h)
        #end
        
        p_h, h = self.forward(_v, dropfrac)
        
        return p_h, _v, p_v
    #end      
    
    
    def train(self, dbn, Xtrain, Ytrain, Xtest, Ytest, train_specs, global_specs):
        '''
        Train procedure of the RBM.
        Here the choice is to associate the gradients to the model itself, so to avoid independent
        variables to the passed to the CD update function. When the model shall be saved, these 
        gradients are deleted, so not to serialize a huge object.
        Note that the training procedure is wholly contained in this block. For the sake of this 
        work this is the best choice
        Input:
            - dbn (dbns.DeepBeliefNetwork): the ``parent'' model which this
                                            RBM comes from. It is useful for saving
                                            the partially-trained DBN
            - Xtrain, Xtest, Ytrain, Ytest (torch.Tensor): dataset
            - train_specs (dict): specifications of the SGD learning algorithm
            - global_specs (dict): global variables
        Output:
            - _Xtrain, _Xtest (torch.Tensor): the dataset projected on the subsequent layer
            
        Note: a crucial aspect of this learning segment is the preparation and update of the
        _Xtrain and _Xtest variables. In the context of the bigger picture involving the DBN
        training, it is important, whether using the greedy or iterative learning strategy,
        to produce, update and swap the datasets in the correct way
        '''
        
        if train_specs['verbose']:
            print('---\nLayer {}: Input size, output size = ({},{})\n'.format(self.layer_id, self.Nv, self.Nh))
        #end
        
        self.dW = torch.zeros_like(self.W)
        self.da = torch.zeros_like(self.a)
        self.db = torch.zeros_like(self.b)
        
        _Xtrain = torch.zeros( [Xtrain.shape[0], Xtrain.shape[1], self.Nh] )
        _Xtest  = torch.zeros( [Xtest.shape[0],  Xtest.shape[1], self.Nh] )
        
        # initialization of the loss function.
        # In the scope of this function, we are training one RBM layer
        # hence it is legitimate to reset gradients, loss, ...
        criterion = MSELoss(reduction = 'mean')
        
        for epoch in range(train_specs['epochs']):
            if train_specs['verbose']: print('\nEpoch {}'.format(epoch))
            
            # mask vector for dropout implementation. Generated at each epoch
            self.dropmask = torch.Tensor(np.random.binomial(1, p = train_specs['dropout'], size = (Xtrain.shape[1], self.Nh)), device = self.device)
            
            train_loss = 0.0
            
            # shuffle the indices to present the data batches in different orders
            batch_indices = list(range(Xtrain.shape[0]))
            random.shuffle(batch_indices)
            for n in batch_indices:
                
                pos_v  = Xtrain[n,:,:]
                
                # parameters update and loss computation
                loss = self.contrastive_divergence_params_update(criterion, pos_v, epoch, Xtrain.shape[1], train_specs)
                train_loss += loss
                # _Xtrain[n,:,:] = self.forward(Xtrain[n,:,:].clone())[0].clone()
                
                # if train_specs['prog_train']:
                #     # if we need to keep track of the performance of the DeltaRule classifier during
                #     # training, we can train it here
                #     linclass_loss, linclass_acc = self.linclass.train(self.forward(pos_v)[0].clone(), Ytrain[n,:,:].clone())
                #     self.loss_finer.append(linclass_loss)
                #     self.acc_finer.append(linclass_acc)
                # #end
                
                # Data forth-projection AFTER parameters update
                _Xtrain[n,:,:], _ = self.forward(pos_v)
                
            #end batches loops
            
            # test set projection
            for n in range(Xtest.shape[0]): 
                _Xtest[n,:,:], _  = self.forward(Xtest[n,:,:])
            #end
            
            self.cost_profile.append(train_loss / Xtrain.shape[0])
            if global_specs['readout_acc']:
                # if we need readout, here we can compute it from the projected activities
                self.readout_profile.append(self.linear_readout(_Xtrain, Ytrain, _Xtest, Ytest))
            #end
            
            if train_specs['verbose']: 
                print('Training loss = {:.6f}'.format(train_loss / Xtrain.shape[0]))
                if global_specs['readout_acc']: 
                    print('[RidgeClassifier] (Test)  Read-out accuracy at layer {:d}: {:.4f}'.format(self.layer_id, self.readout_profile[-1]))
                #end
            #end
            
            if global_specs['save_tmp']:
                # we may want to save the partially trained model, at an
                # epoch stamp lesser than the whole training span
                if epoch in global_specs['epochs_glws']:
                    dbn.save_model(global_specs, epoch = epoch, rbml = self.layer_id)
                #end
            #end
            
        #end epochs loop
        
        return _Xtrain, _Xtest
    #end
    
    
    def contrastive_divergence_params_update(self, criterion, pos_v, epoch, batch_size, train_specs):
        '''
        Approximation of the likelihood gradients. The Contrastive Divergence gradients are used instead.
        Input:
            - criterion (torch.nn.MSELoss): loss function (PyTorch buildin class), to compute the 
                                            reconstruction error
            - pos_v (torch.Tensor): visible activities
            - epoch (int): epoch stamp. So to switch the momentum value
            - batch_size (int): batch size, to average out the CD gradients
            - train_specs (dict): to pass compactly the hyper-params
        Output:
            - loss (float): value of the reconstruction error
        '''
        
        momentum = train_specs['initial_momentum']
        lr       = train_specs['initial_lr']
        penalty  = train_specs['weight_decay']
        
        if train_specs['binarize_act']: pos_v = torch.bernoulli(pos_v.clone())
        
        pos_ph, pos_h = self.forward(pos_v, train_specs['dropout'])
        neg_ph, neg_v, neg_pv = self.Gibbs_sampling(pos_v, train_specs['mcmc_steps'], train_specs['dropout'])
        # if train_specs['binarize_act']: neg_h = torch.bernoulli(neg_ph)
        
        pos_dW = matmul(pos_v.t(), pos_ph).t() / batch_size
        pos_da = torch.sum(pos_v, dim = 0) / batch_size 
        pos_db = torch.sum(pos_ph, dim = 0) / batch_size
        
        neg_dW = matmul(neg_v.t(), neg_ph).t() / batch_size
        neg_da = torch.sum(neg_v, dim = 0) / batch_size 
        neg_db = torch.sum(neg_ph, dim = 0) / batch_size
        
        if epoch >= 5: momentum = train_specs['final_momentum']
        
        self.dW = momentum * self.dW + lr * ((pos_dW - neg_dW) - penalty * self.W) 
        self.da = momentum * self.da + lr * (pos_da - neg_da)
        self.db = momentum * self.db + lr * (pos_db - neg_db)
        
        self.W = self.W + self.dW
        self.a = self.a + self.da
        self.b = self.b + self.db
        
        loss = criterion(pos_v, neg_pv)
        return loss.item()
    #end
    
    
    @staticmethod
    def linear_readout(Xtrain, Ytrain, Xtest, Ytest):
        '''
        Readout (accuracy) evaluation. To assess the uniqueness of the projected patterns.
        A ridge classifier is used.
        Input:
            - Xtrain, ... (torch.Tensor): dataset. Note the Xtrain and Xtest are the 
                                          projected values. Y labels do not need to change.
        Output:
            - accuracy_score (float): accuracy of the classification of the given data.
        '''
        
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import accuracy_score
        
        num_batches_train = Xtrain.shape[0]
        batch_size_train  = Xtrain.shape[1]
        train_set_length  = num_batches_train * batch_size_train
        Xtrain            = Xtrain.cpu().numpy().reshape(train_set_length, -1)
        Ytrain            = Ytrain.cpu().numpy().reshape(train_set_length, -1).ravel()
        
        num_batches_test  = Xtest.shape[0]
        batch_size_test   = Xtest.shape[1]
        test_set_length   = num_batches_test * batch_size_test
        Xtest             = Xtest.cpu().numpy().reshape(test_set_length, -1)
        Ytest             = Ytest.cpu().numpy().reshape(test_set_length, -1).ravel()
        
        classifier = RidgeClassifier()
        classifier.fit(Xtrain, Ytrain)
        predicted_labels = classifier.predict(Xtest)
        
        return accuracy_score(Ytest, predicted_labels)
    #end
    
 #endCLASS
 
 
 
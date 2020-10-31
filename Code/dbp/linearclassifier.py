

'''
This module contains the classes used in the numerosity discrimination or estimation tasks,
with the SZ trained DBN.
'''

import torch
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd


class DeltaRuleclassifier:
    '''
    Delta rule (binary) classificator.
    Here a binary classificator is the best choice since the numerosity discrimination task
    requires the DBN to ``recognize'' patterns with a greater or smaller number of items displayed,
    depending on the reference number.
    '''
    
    def __init__(self, Nin, Nout, Nref, hyperparams, discrimination_ranges):
        '''
        Initialization.
        Input:
            - Nin, Nout (int): dimensions
            - Nref (int): reference number to discriminate the patterns against
            - lr , ... (float): hyperparameters
        Output:
            none
        '''
        
        self.Nref = Nref
        self.rrange = discrimination_ranges
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.W = torch.normal(0, 1, (Nin+1, Nout)) * 0.1
        self.loss_metric = torch.nn.MSELoss(reduction = 'mean')
        self.losses = list()
        
        self.lr = hyperparams['learning_rate']
        self.wd = hyperparams['weights_decay']
        self.momentum_final = hyperparams['momentum_final']
        self.momentum_initial = hyperparams['momentum_initial']
        
        self.dw_velocity = torch.zeros_like(self.W)
    #end
    
    '''
    Momentum and previous gradients accessor methods
    '''
    def getMomentum(self, epoch):
        if epoch >= 5:
            return self.momentum_final
        else:
            return self.momentum_initial
        #end
    #end
    
    def getVelocity(self):
        return self.dw_velocity
    #end
    
    def setVelocity(self, velocity):
        self.dw_velocity = velocity
    #end
    
    def resetVelocity(self):
        self.dw_velocity = torch.zeros_like(self.W)
    #end
    
    def sigmoid(self, x):
        '''
        Compute the predictions
        Input:
            - x (torch.Tensor): input values
        Output:
            - y (torch.Tensor): predictions
        '''
        return 1/(1 + torch.exp( -torch.matmul(x, self.W) ))
    #end
    
    def predict(self, x):
        return self.sigmoid(x)
    #end
    
    def train(self, Xtr, Ytr, global_specs):
        '''
        Train the delta rule.
        Input:
            - Xtr, Ytr (torch.Tensor): data set
            - global_specs (dict): global variables
        Output:
            none
        '''
        
        losses = np.zeros(global_specs['lc_train_epochs'])
        accs = np.zeros(global_specs['lc_train_epochs'])
        
        # this re-batching strategy serves the purpose of 
        # speeding up the DR training. Of course, if the 
        # single batch strategy is used, the classifier 
        # converges very slowly, if at all
        if global_specs['dataset'] == 'mnist':
            nb = 60; bs = 1000
        elif global_specs['dataset'] == 'sz':
            nb = 10; bs = 5120
        #end
        
        X = Xtr.view(nb, bs, -1)
        Y = Ytr.view(nb, bs, -1)
        
        sys.stdout.flush()
        for i in tqdm(range(global_specs['lc_train_epochs'])):
            
            train_loss = 0.0
            train_acc = 0.0
            
            for n in range(X.shape[0]):
                
                loss, out = self.fit(X[n,:,:], Y[n,:,:])
                acc = self.getBinaryAccuracy(out.clone(), Y[n,:,:].clone())
                train_loss += loss
                train_acc += acc
            #end
            
            losses[i] = train_loss / X.shape[0]
            accs[i] = train_acc / X.shape[0]
        #end
        
        global_specs['metrics'][global_specs['web_nref']].at[global_specs['current_epoch'], 'TrainAcc'] = accs[-1]
        global_specs['metrics'][global_specs['web_nref']].at[global_specs['current_epoch'], 'TrainLoss'] = losses[-1]
        # self.performance_vis(losses, accs, global_specs)
    #end
    
    
    def fit(self, x, y, epoch = 0):
        '''
        Update step for the classifier parameters
        Input:
            - x, y (torch.Tensor): data batches
            - epoch (int): epoch time stamp, to switch the 
                           momentum value
        Output:
            - error (float): MSE loss
            - out (torch.Tensor): predictions
        '''
        
        x_ = torch.cat((torch.ones(x.shape[0], 1), x), 1)
        y_ = self.binarize_labels(y.clone())
        
        out = self.sigmoid(x_)
        out = self.binarize_pred(out)
        
        momentum = self.getMomentum(epoch)
        velocity = self.getVelocity()
        
        weights_change = torch.matmul(x_.t(), (y_ - out)) # this is delta rule, then we add 
        final_weights_change = momentum * velocity + self.lr * (weights_change - self.wd * self.W)
        self.W = self.W + final_weights_change
        self.W = self.W + weights_change * self.lr
        
        self.setVelocity(final_weights_change)
        
        error = self.loss_metric(y_, out).item()
        return error / x_.shape[0], out
    #end
    
    def getBinaryAccuracy(self, out, y):
        '''
        Get the value of accuracy, true labels against predictions
        Input:
            - out (torch.Tensor): predictino
            - y (torch.Tensor): true labels
        Output:
            acc (float): accuracy
        '''
        
        assert out.shape[0] == y.shape[0], 'SIZES MISMATCH'
        out = self.binarize_pred(out.clone())
        y_ = self.binarize_labels(y.clone())
        return torch.sum(out == y_).float() / y.shape[0]
    #end
    
    def getDiscriminationResults(self, x, y):
        '''
        Takes the whole window of useful numbers, given a reference,
        and for each one of these, the discrimination accuracies are computed
        Input:
            - x, y (torch.Tensor): data set
            - alg (str): algorithm identifier
            - path (str): path to save the data structures obtained, if it could be useful
        Output:
            - ratios (list): numerical ratios between number in the proper window and reference number
                             E.g., if Nref = 8, window = [5,6,...,12], ratios = [5/8, 6/8, ... , 12/8]
            - percs (list): percentages of samples having label i \in window that are correctly classified 
                            That is, given the correct label `greater than' or `smaller than' the reference number
        '''
        
        x = x.view(-1, x.shape[-1])
        y = y.view(-1,1)
        _filter = ((y >= self.rrange[self.Nref][0]) & (y <= self.rrange[self.Nref][-1])).view(-1)
        xte = x[_filter]
        yte = y[_filter]
        
        out = self.predict( torch.cat((torch.ones(xte.shape[0], 1), xte), 1))
        ratios = list()
        percs = list()
        
        for i in self.rrange[self.Nref]:
            
            mask = (torch.tensor(i) == yte)
            pred = self.binarize_pred(out[mask])
            ratios.append(i / self.Nref)
            percs.append( torch.sum(pred) / pred.shape[0] )
        #end
        
        return ratios, percs
    #end
    
    def binarize_pred(self, out):
        '''
        Binarize the predictions. From probabilities to binary values
        '''
        out[out < 0.5] = torch.tensor([0.]).to(self.device)
        out[out > 0.5] = torch.tensor([1.]).to(self.device)
        return out
    #end
    
    def binarize_labels(self, y):
        '''
        Binarize the labels. 
        E.g., if the number of reference is Nref = 8, then all the labels
        lesser than 8 are set to 0, while the greater ones are set to 1
        '''
        y[y <= self.Nref] = torch.tensor([0.]).to(self.device)
        y[y > self.Nref]  = torch.tensor([1.]).to(self.device)
        return y
    #end
    
    @staticmethod
    def performance_vis(losses, accs, global_specs):
        '''
        If necessary, visualize the performance of the classifier
        Input:
            - losses (list or list-like): values of the loss during training
            - accs (list or list-like): values of the accuracies during training
            - global_specs (dict): global variables
        Output:
            none
        '''
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
        ax.plot(np.arange(0, losses.size), losses, color = 'r', lw = 1.5, alpha = 0.75)
        ax.set_xlabel('Epochs', fontsize = 14)
        ax.set_ylabel('MSE', color = 'r', fontsize = 14)
        ax.tick_params(axis = 'y', labelcolor = 'r')
        axt = ax.twinx()
        axt.plot(np.arange(0, accs.size), accs, color = 'g', lw = 1.5, alpha = 0.75)
        axt.set_ylabel('Accuracy', color = 'g', fontsize = 14)
        axt.tick_params(axis = 'y', labelcolor = 'g')
        # fig.savefig(global_specs['path_images'] + r'/{}/profiles_and_errors/dr_fare_{}_{}_N{}.png'.format(global_specs['scheme'], global_specs['alg'], global_specs['current_epoch'],                         global_specs['web_nref']), 
        #             format = 'png', bbox_inches = 'tight', dpi = 300)
        plt.show(fig)
        plt.close(fig)
    #end
#end



class MLPclassifier:
    '''
    MLP classifier, used in the case of numerosity estimation.
    Unlike the case of the DeltaRule classifier, here the model should produce an estimate of 
    the true label associated with the input pattern.
    NOTE: The hidden layers of the DBN show patterns of activity that, is training is properly made,
    should be univoque representations of the input data in the space spanned by the hidden units. 
    Hence a shallow network should suffice to classify these activities.
    '''
    def __init__(self, Nin, Nout, global_specs):
        '''
        Initialization. 
        Input:
            - Nin (int): number of input features
            - Nout (int): output units, depending on the dataset
        Output:
            none
        '''
        
        self.classes = global_specs['data_classes']
        self.dataset = global_specs['dataset']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.W = torch.normal(0, 1, (Nout, Nin)) * 0.1
        self.b = torch.zeros(1, Nout)
        self.output = torch.nn.Softmax(dim = 1)
        self.loss_metric = torch.nn.MSELoss(reduction = 'mean')
        
        self.lr = global_specs['lc_hyperparams']['learning_rate']
        self.wd = global_specs['lc_hyperparams']['weights_decay']
        self.momentum_initial = global_specs['lc_hyperparams']['momentum_initial']
        self.momentum_final = global_specs['lc_hyperparams']['momentum_final']
        
        self.resetVelocity()
    #end
    
    '''
    Momentum and previous gradients accessor methods
    '''
    def getMomentum(self, epoch):
        if epoch >= 5:
            return self.momentum_final
        else:
            return self.momentum_initial
        #end
    #end
    
    def getVelocity(self):
        return self.vel_W, self.vel_b
    #end
    
    def setVelocity(self, vel_W, vel_b):
        self.vel_W = vel_W
        self.vel_b = vel_b
    #end
    
    def resetVelocity(self):
        self.vel_W = torch.zeros_like(self.W)
        self.vel_b = torch.zeros_like(self.b)
    #end
    
    def fit(self, X, Y, epoch = 0):
        '''
        Single parameters update given the input data.
        Input:
            - X, Y (torch.Tensor): input data and associated labels, training set
            - epoch (int): number of the training time stamp, used to switch the momentum value
        Output:
            - loss (torch.Tensor): scalar value of the loss associated with the current prediction
            - out (torch.Tensor): output prediction
        '''
        
        out = self.output(torch.matmul(X, self.W.t()) + self.b)
        pred_oh = self.label2onehot(self.onehot2label(out)).float()
        true_oh = self.label2onehot(Y).float()
        
        '''
        NOTE: due to the one-hot convention --see below-- we must trim the first 
        column of the labels matrix.
        '''
        err = (pred_oh - true_oh)[:,1:]
        
        grad_W = torch.matmul(err.t(), X) / X.shape[0]
        grad_b = err.sum(dim = 0) / X.shape[0]
        
        momentum = self.getMomentum(epoch)
        vel_W, vel_b = self.getVelocity()
        
        dW = momentum * vel_W - self.lr * grad_W - self.wd * self.W
        db = momentum * vel_b - self.lr * grad_b
        self.W = self.W + dW
        self.b = self.b + db
        
        self.setVelocity(dW, db)
        
        return self.loss_metric(pred_oh, true_oh).item(), out
    #end
    
    def train(self, Xtrain, Ytrain, global_specs):
        '''
        Training procedure.
        Input:
            - Xtrain, Ytrain (torch.Tensor): training set
            - global_specs (dict): global variables
        Output:
            none
            
        Note that, to speed up training, a rebatching strategy is adopted.
        The values of batch size and number of batches is hardcoded, but could be
        specified in the global variables
        '''
        
        if self.dataset == 'mnist':
            nb = 60; bs = 1000
        elif self.dataset == 'sz':
            nb = 10; bs = 5120
        #end
        
        Xtrain = Xtrain.view(nb, bs, Xtrain.shape[-1])
        Ytrain = Ytrain.view(nb, bs, -1)
        
        lloss = np.zeros(global_specs['lc_train_epochs'])
        lacc  = np.zeros(global_specs['lc_train_epochs'])
        
        sys.stdout.flush()
        for i in tqdm(range(global_specs['lc_train_epochs'])):
            
            trloss = 0.0
            tracc  = 0.0
            for n in range(Xtrain.shape[0]):
                
                loss, out = self.fit(Xtrain[n,:,:], Ytrain[n,:,:], epoch = i)
                trloss += loss
                tracc += self.getAccuracy(out, Ytrain[n,:,:])
            #end
            
            trloss = trloss / Xtrain.shape[1]
            tracc = tracc / Xtrain.shape[0]
            lloss[i] = trloss
            lacc[i] = tracc
        #end
        
        # self.performance_vis(lloss, lacc, global_specs)
    #end
    
    def getAccuracy(self, out, Y):
        '''
        Get the accuracy associated with a prediction.
        Note that, in the case of the SZ dataset, the trend of accuracy could
        show himself as frustratingly unsatisfactorily. But neither one should expect a clean
        trend as in the case of MNIST dataset. In this latter case, the features learned by 
        the network are more ``regular'' and reusable across the different classes. Eg, the 
        curvature of the 0 digit could be seen also in the 9 or 8 digits. 
        In this case, the classes are not independent. In the case of the SZ dataset, on the other
        hand, classes are independent. The numerosity encoded by the activation given by a given datum
        could share few with the activation caused by a different input. 
        Even if the classifier predicts a ``8'' value for an input with 9 items in it, or predicts ``31''
        for a pattern with 30 items, the performance should be considered good enough. 
        Further analyses, such as the visualization of the numerosity estimation results, could show that
        the performance of the network behaves in a manner similar to the human benchmarks.
        Input:
            - out (torch.Tensor): predictions
            - Y (torch.Tensor): true labels
        Output:
            - accuracy (torch.Tensor): scalar value of the accuracy
        '''
        
        assert out.shape[0] == Y.shape[0], 'SIZES MISMATCH'
         
        # out[out > 0.5] = 1
        # out[out <= 0.5] = 0
        pred_idx = out.argmax(dim = 1).view(Y.shape[0], -1)
        true_idx = Y.argmax(dim = 1).view(Y.shape[0], -1)
        
        # pred_idx = torch.argmax(out, 1)
        return torch.sum(pred_idx.view(1,-1) == true_idx.view(1,-1)).float() / Y.shape[0]
    #end
    
    
    def label2onehot(self, Y):
        '''
        Encode labels as one-hot vectors. 
        Here one should pay particular attention to the classes labels: does the dataset 
        represent objects that could be indicized as {0, ..., N}, or, like in the case of 
        numerosity estimation, do we want the labels to be {1, ..., N}? 
        In the former case (MNIST), this code should be further refined. 
        This version support the convention {1, ..., N}, hence the one-hots matrix is generated 
        having as columns the number of classes plus one, so to be able to encode the 32 value.
        Input:
            - Y (torch.Tensor): actual labels
        Output:
            - Yoh (torch.Tensor): one-hot labels
        '''
        
        Y = Y.type(torch.int64)
        nrows = Y.shape[0]
        ncols = self.classes
        Yoh = torch.zeros((nrows, ncols+1), dtype = torch.int64)
        Yoh.scatter_(1, Y, 1)
        
        return Yoh
    #end
    
    @staticmethod
    def onehot2label(Yoh):
        
        return (Yoh.argmax(dim = 1) + 1).view(-1,1)
    #end
    
    
    def numerosity_estimation(self, Xtest, Ytest, global_specs):
        '''
        For each label, that is, for each numerosity (in this case), we select
        all and only the patterns that have this label associated. We then require
        the model to predict the related output, and we compute the mean and standard
        deviation of the answers. In this way, we can assess the accuracy of the 
        numerosity estimation task.
        Input:
            - Xtest, Ytest (torch.Tensor): test set
            - global_specs (dict): global variables
        Output:
            psydata (pandas.DataFrame): data frame containing the estimation results
        '''
        
        Ytest = Ytest.type(torch.int64).view(-1, Ytest.shape[-1])
        Xtest = Xtest.view(-1, Xtest.shape[-1])
        psydata = pd.DataFrame(index = global_specs['estimation_range'], columns = ['n', 'std'])
        indices = global_specs['estimation_range']
        
        for i in indices:
            _filter = (Ytest == i).view(-1)
            _Xtest = Xtest[_filter]
            _out = self.output(torch.matmul(_Xtest, self.W.t()) + self.b)
            out  = self.onehot2label(_out)
            psydata.at[i, 'n'] = out.float().mean().cpu().numpy()
            psydata.at[i, 'std'] = out.float().std().cpu().numpy()
        #end
        
        return psydata
    #end
    
    def test(self, X, Y):
        '''
        Could be useful to get the test set accuracy of the model.
        Input:
            - X, Y (torch.Tensor): test set
        Output:
            none
        '''
        
        N = X.shape[0]
        acc = 0.0
        for n in range(X.shape[0]):
            
            _, out = self.forward(X[n,:,:], Y[n,:,:])
            acc += self.getAccuracy(out, Y[n,:,:])
        #end
        print('\nOverall accuracy = {:2f}'.format(acc / N))
    #end
    
    
    @staticmethod
    def performance_vis(losses, accs, global_specs):
        '''
        Could be useful to see the classifier performance. 
        As above
        '''
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
        ax.plot(np.arange(0, losses.size), losses, color = 'r', lw = 1.5, alpha = 0.75)
        ax.set_xlabel('Epochs', fontsize = 14)
        ax.set_ylabel('MSE', color = 'r', fontsize = 14)
        ax.tick_params(axis = 'y', labelcolor = 'r')
        axt = ax.twinx()
        axt.plot(np.arange(0, accs.size), accs, color = 'g', lw = 1.5, alpha = 0.75)
        axt.set_ylabel('Accuracy', color = 'g', fontsize = 14)
        axt.tick_params(axis = 'y', labelcolor = 'g')
        # fig.savefig(global_specs['path_images'] + r'/{}/profiles_and_errors/dr_fare_{}_{}_N{}.png'.format(global_specs['scheme'], global_specs['alg'], global_specs['current_epoch'], global_specs['web_nref']), 
        #             format = 'png', bbox_inches = 'tight', dpi = 300)
        plt.show(fig)
        plt.close(fig)
    #end
#end





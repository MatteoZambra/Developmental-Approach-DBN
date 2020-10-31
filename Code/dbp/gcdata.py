
'''
Class DataLoad and derivatives
------------------------------

Used to download/create/fetch the MNIST dataset, using the PyTorch data utilities.
Or, in alternative, to create the SZ dataset, using the .mat files available at http://ccnl.psy.unipd.it/research/deeplearning

Note: to use the code without having the MNIST dataset saved on yout machine, it sufficies to create a directory called `dataset' 
in the same location of the main.py script (or MainScript.ipynb notebook). The program flow will download the dataset and keep and 
use it in the following steps.
The `dataset' directory is necessary also in the case of the SZ dataset. There should be the SZ_data.mat and SZ_data_test.mat data.
'''

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np



class DataLoad:
    '''
    Parent class in which the initialization and accesor methods are placed
    '''
    
    def __init__(self, data_specs):
        '''
        Initialization
        Input:
            - data_specs (dict): data managing specifications
        Output:
            none
        '''
        
        self.device     = data_specs['device']
        self.batch_size = data_specs['batch_size']
    #end
    
    #getters
    def GetNfeats(self):
        return self.nfeat
    
    def GetXtrain(self):
        return self.Xtrain.clone()
    
    def GetXtest(self):
        return self.Xtest.clone()
    
    def GetYtrain(self):
        return self.Ytrain.clone()
    
    def GetYtest(self):
        return self.Ytest.clone()
    
    def getLabelsRange(self):
        y = self.Ytest
        y = np.array(y.numpy(), dtype = np.int64).flatten()
        labels_list = list(set(list(y)))
        return labels_list
    #end getters
#end


class MNISTdata(DataLoad):
    '''
    Uses the PyTorch data utilities.
    The dataset is formatted in the following way:
        [num_batches, batch_size, num_features]
    In the case of test set, the last dimension is 1 since
    there we have the labels
    '''

    def __init__(self, data_specs):
        '''
        Initialization.
        Input:
            - data_specs (dict): see above
        Output:
            none
        
        Note that here the user can choose among the creation or fetching of pre-created data.
        If the choice is `create', then the dataset is downloaded.
        If the choice is to `save' the dataset, then a copy is saved in the `dataset' directory.
        '''
        
        self.transforms = [transforms.ToTensor()]
        super().__init__(data_specs)
        
        if data_specs['stream'] == 'fetch':
            with open(data_specs['path'] + r'/dataset_tensor_bs{}_{}.pickle'.format(self.batch_size,
                                                                               data_specs['binarize']), 'rb') as handle: 
                dataset = torch.load(handle, map_location = self.device)
            #end
            
            self.Xtrain = dataset['train_images']
            self.Xtest  = dataset['test_images']
            self.Ytrain = dataset['train_labels']
            self.Ytest  = dataset['test_labels']
            
            self.nfeat = self.Xtrain.shape[2]
            print('Data fetched')
            
        elif data_specs['stream'] == 'create':
            self.yield_tensor_data(data_specs)
        #end
        
    #end
    
    
    def yield_tensor_data(self, data_specs):
        '''
        Given the choice of downloading the dataset, here it is formatted and transposed
        in an amenable format, see above.
        Input:
            - data_specs (dict): see above
        '''
        
        transfs = transforms.Compose(self.transforms)
        
        train_data = MNIST(r'data/', download = True, train = True,  transform = transfs)
        test_data  = MNIST(r'data/', download = True, train = False, transform = transfs)
        
        train_load = DataLoader(train_data, batch_size = self.batch_size, shuffle = False)
        test_load  = DataLoader(test_data,  batch_size = self.batch_size, shuffle = False)
        
        train_iterator = list(iter(train_load))
        test_iterator  = list(iter(test_load))
        
        self.nfeat   = train_iterator[0][0].shape[2] * train_iterator[0][0].shape[3]
        self.nbtrain = train_iterator.__len__()
        self.nbtest  = test_iterator.__len__()
        
        Xtrain = torch.zeros([self.nbtrain, self.batch_size, self.nfeat]).to(self.device)
        Xtest  = torch.zeros([self.nbtest, self.batch_size, self.nfeat]).to(self.device)
        Ytrain = torch.zeros([self.nbtrain, self.batch_size, 1], dtype = torch.int32).to(self.device)
        Ytest  = torch.zeros([self.nbtest, self.batch_size, 1], dtype = torch.int32).to(self.device)
        
        for n in range(self.nbtrain):
            data, labels  = train_iterator[n]
            Xtrain[n,:,:] = data.view(self.batch_size, self.nfeat).to(self.device)
            Ytrain[n,:,:] = labels.view(self.batch_size, 1).to(self.device)
        #end
        
        for n in range(self.nbtest):
            data, labels = test_iterator[n]
            Xtest[n,:,:] = data.view(self.batch_size, self.nfeat).to(self.device)
            Ytest[n,:,:] = labels.view(self.batch_size, 1).to(self.device)
        #end
        
        if data_specs['binarize'] == 'b':
            # if we want to binarize the dataset, and the threshold factor 
            # should be specified in the main program
            
            threshold = 1/data_specs['factor']
            Xtrain[Xtrain <= threshold] = 0.0
            Xtrain[Xtrain >  threshold] = 1.0
            Xtest[Xtest <= threshold] = 0.0
            Xtest[Xtest >  threshold] = 1.0
        #end
        
        self.Xtrain = Xtrain
        self.Xtest  = Xtest
        self.Ytrain = Ytrain
        self.Ytest  = Ytest
        
        if data_specs['save']:
            with open(data_specs['path'] + r'/dataset_tensor_bs{}_{}.pickle'.format(self.batch_size, 
                                                                                      data_specs['binarize']), 'wb') as handle: 
                torch.save({'train_images' : Xtrain, 'train_labels' : Ytrain,
                            'test_images'  : Xtest,  'test_labels'  : Ytest}, handle)
            #end
            handle.close()
        #end
        
        print('Data created and saved')
    #end
    
#endCLASS


class SZdata(DataLoad):
    '''
    Data class that uses the files available at http://ccnl.psy.unipd.it/research/deeplearning
    Here it is not necessary to choose whether to download or fetch the data. The dataset is 
    created at each run from these fils
    '''
    
    def __init__(self, data_specs):
        '''
        Initialization.
        Input:
            - data_specs (dict): as above
        Output:
            none
        '''
                
        self.device = data_specs['device']
        self.batch_size = data_specs['batch_size']
        super().__init__(data_specs)
        
        Xtrain, Ytrain = self.yield_tensor_data(data_specs['path'] + r'/SZ_data.mat', kind = 'batch')
        Xtest, Ytest = self.yield_tensor_data(data_specs['path'] + r'/SZ_data_test.mat', kind = 'whole')
        
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest
    #end
    
    
    def yield_tensor_data(self, path, kind):
        '''
        Creation of the dataset in the same format as above, that is 
            [ num_batches, batch_size, num_features ]
        Input:
            - path (str): path where the raw data lay
            - data_specs (dict): as above
        Output:
            - X, Y (torch.Tensor): dataset properly formatted
        '''
        
        from scipy.io import loadmat
        import numpy as np
        
        dataset = loadmat(path)
        Xd = dataset['D'].T
        Yd = dataset['N_list'].T
        
        N = Xd.shape[0]
        
        idx = np.random.permutation(N)
        Xd = Xd[idx]
        Yd = Yd[idx]
        
        self.nfeat = Xd.shape[1]
        
        if kind == 'batch':
            
            self.nb = int(N / self.batch_size)
            
            X = torch.zeros([self.nb, self.batch_size, self.nfeat]).to(self.device)
            Y = torch.zeros([self.nb, self.batch_size, 1]).to(self.device)
            
        elif kind == 'whole':
            
            self.nb = 1
            self.batch_size = Xd.shape[0]
            
            X = torch.zeros([self.nb, self.batch_size, self.nfeat]).to(self.device)
            Y = torch.zeros([self.nb, self.batch_size, 1]).to(self.device)
            
        else:
            raise RuntimeError('No batching strategy provided')
        #end
        
        for n in range(self.nb):
            idx = n * self.batch_size
            X[n,:,:] = torch.Tensor(Xd[idx : idx + self.batch_size]).to(self.device)
            Y[n,:,:] = torch.Tensor(Yd[idx : idx + self.batch_size]).to(self.device)
        #end
        
        return X, Y
    #end       
        
#end
        
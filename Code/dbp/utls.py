'''
Utilities module
'''


import torch
from torch import sigmoid, matmul
from torch.nn import MSELoss

import numpy as np
import gcvis

import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')



def fetch(path):
    '''
    Load the object pointed by path
    Input:
        - path (str): path of the object to load
    Output:
        - dbn (dbns.DeepBeliefNetwork): DBN model previously learned and saved to disk
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(path, 'rb') as f:
        dbn = torch.load(f, map_location = device)
    #end
    f.close()
    
    return dbn
#end




'''
Performance evaluation functions:
    - MSE_averages calls assessment
    - assessment calls forward
'''

def forward(Xtest, Ytest, dbn, plot_specs, mode):
    '''
    Propagation forth and back of the test set, so to have available the 
    reconstruction of the samples, in order to compute the reconstruction error.
    This function is called by the function assessment for each one of the 
    reconstruction tasks, and for each of these the reconstruction error could
    be computed.
    Input:
        - Xtest, Ytest (torch.Tensor): test set
        - dbn (dbns.DeepBeliefNetwork): model
        - mode (str): either to reconstruct simply or corrupt the
                      image with occlusions or noise N(0,0.5)
        - plot_specs (dict): contains the general purpose istruction for plotting
    Output:
        - _altered (torch.Tensor): corrupted data
        - _Xtest (torch.Tensor): reconstructed data
        - test_loss (float): scaled for the number of batches. Reconstruction error
    '''
    
    num_batches = Xtest.shape[0]
    _Xtest = torch.zeros_like(Xtest)  
    _altered = torch.zeros_like(Xtest)
    
    criterion = MSELoss(reduction = 'mean')
    test_loss = 0.0
    
    for m in range(num_batches):
        
        _v = Xtest[m,:,:].clone()
        
        if mode == 'rep':
            _v = Xtest[m,:,:].clone()
            _altered[m,:,:] = _v.clone()
        
        elif mode == 'rec':
            side_dim     = int(np.sqrt(Xtest[0].shape[1]))
            row          = 5
            rows_to_kill = 5
            
            _v[:, (row + 1)*side_dim : (row + rows_to_kill)*side_dim + side_dim] = 0.0
            _altered[m,:,:] = _v.clone()
            
        elif mode == 'den':
            if torch.cuda.is_available():
                _v = _v + torch.normal(0, 0.5, Xtest[m,:,:].shape).cuda()
            else:
                _v = _v + torch.normal(0, 0.5, Xtest[m,:,:].shape)
            #end
            _altered[m,:,:] = _v.clone()
            
        #end
           
        else:
            raise Exception('No match for mode')
        #end
        
        for k in range(plot_specs['loops']):
            for layer in range(dbn.nlayers):
                _v = sigmoid(matmul(_v, dbn.rbm_layers[layer].W.t()) + dbn.rbm_layers[layer].b)
            #end
            _reconstr = _v.clone()
            
            for layer in range(dbn.nlayers-1, -1,-1):
                _reconstr = sigmoid(matmul(_reconstr, dbn.rbm_layers[layer].W) + dbn.rbm_layers[layer].a)
            #end
        #end
        
        _true_image = Xtest[m,:,:].clone()
        loss = criterion(_true_image, _reconstr)
        test_loss += loss.item()
        
        _Xtest[m,:,:] = _reconstr
    #end
    
    return _altered, _Xtest, test_loss / num_batches
#end


def assessment(Xtest, Ytest, dbn, run, plot_specs):
    '''
    Recreation of the data samples (in the case of the MNIST dataset, the images could be corrupted)
    and computation of the reconstruction errors at the end of training.
    This function serves the purpose of testing the learned DBN.
    Input:
        - Xtest, Ytest (torch.Tensor): test set
        - dbn (dbns.DeepBeliefNetworks): model to test
        - run (int): plot the digits only for the first run, otherwise the visualization becomes heavy
        - plot_specs (dict): see above
    Output:
        - [errors] (list): reconstruction errors (floats) for the various tasks
    '''
    
    plot_specs['algorithm'] = dbn.algorithm
    
    originals, reproduction, mse_rep   = forward(Xtest, Ytest, dbn, plot_specs, mode = 'rep')
    corrupted, reconstruction, mse_rec = forward(Xtest, Ytest, dbn, plot_specs, mode = 'rec')
    noised, denoised, mse_den          = forward(Xtest, Ytest, dbn, plot_specs, mode = 'den')
    
    # plot
    if run == 0 and plot_specs['plot_digits']:
        print('\n\nTrain {}'.format(plot_specs['algorithm']))
        # indices = [np.random.randint(0, len(Xtest)) for _ in range(1)]
        indices = [0, 1, 5, 9]
        
        for idx in indices:
            if plot_specs['dataset'] == 'sz':
                # owing to the different format of the dataset
                gcvis.plot_images_grid_save(originals.view(512,100,-1)[idx], Ytest.view(512,100,-1)[idx], 'Original samples', idx, plot_specs, mode = 'original')
                gcvis.plot_images_grid_save(reproduction.view(512,100,-1)[idx], Ytest.view(512,100,-1)[idx], 'Reconstructed samples', idx, plot_specs, mode = 'reproduction')
            elif plot_specs['dataset'] == 'mnist':
                gcvis.plot_images_grid_save(originals[idx], Ytest[idx], 'Original samples', idx, plot_specs, mode = 'original')
                gcvis.plot_images_grid_save(reproduction[idx], Ytest[idx], 'Reconstructed samples', idx, plot_specs, mode = 'reproduction')
                gcvis.plot_images_grid_save(corrupted[idx], Ytest[idx], 'Corrupted samples', idx, plot_specs, mode = 'corrupted')
                gcvis.plot_images_grid_save(reconstruction[idx], Ytest[idx], 'Reconstructed samples', idx, plot_specs, mode = 'recreated')
                gcvis.plot_images_grid_save(noised[idx], Ytest[idx], 'Noised samples', idx, plot_specs, mode = 'noised')
                gcvis.plot_images_grid_save(denoised[idx], Ytest[idx], 'Reconstructed samples', idx, plot_specs, mode = 'denoised')
            #end
        #end
    #end
    
    return [mse_rep, mse_rec, mse_den]
#end


def MSE_averages(Xtest, Ytest, plot_specs):
    '''
    The above assessment function is used to collect the run-wise values of 
    reconstruction error. These values will be used to compute the average 
    over the runs specified by the users.
    The task of this function is to collect these run specific values to proper 
    data structures, so to plot them.
    Input:
        - Xtest, Ytest (torch.Tensor): test set
        - plot_specs (dict): see above
    Output:
        none
    '''
    
    runs = plot_specs['runs']
    mse_greedy = np.zeros((runs, 3))
    mse_iterative = np.zeros((runs, 3))
    dropt = '_drop' + str(plot_specs['dropout']).replace('.','d') if plot_specs['dropout'] < 1 else '_nodrop'
    gcorr = '_nogcorr' if plot_specs['gcorrection'] == 1 else '_gcorr' + str(plot_specs['gcorrection']).replace('.', 'd')
    
    for run in range(runs):
        mname = r'/series_{}runs_{}/{}_{}_greedy_dbn{}_cd{}_ep{}{}{}.pkl'.format(plot_specs['runs'], 
                                                                    plot_specs['scheme'], plot_specs['dataset'],
                                                                    plot_specs['scheme'], 
                                                                    run, 
                                                                    plot_specs['mcmc_steps'], 
                                                                    plot_specs['epochs'], 
                                                                    dropt,
                                                                    gcorr
                                                                    )
        with open(plot_specs['path_models'] + mname, 'rb') as f:
            gdbn = torch.load(f, map_location = 'cuda') 
        #end
        f.close()
        
        mseg = assessment(Xtest, Ytest, gdbn, run, plot_specs)
        
        mname = r'/series_{}runs_{}/{}_{}_iterative_dbn{}_cd{}_ep{}{}{}.pkl'.format(plot_specs['runs'], 
                                                                      plot_specs['scheme'], plot_specs['dataset'],
                                                                      plot_specs['scheme'], 
                                                                      run,
                                                                      plot_specs['mcmc_steps'], 
                                                                      plot_specs['epochs'], 
                                                                      dropt,
                                                                      gcorr
                                                                      )
        with open(plot_specs['path_models'] + mname, 'rb') as f:
            idbn = torch.load(f, map_location = 'cuda' if torch.cuda.is_available() else 'cpu') 
        #end
        f.close()
        
        msei = assessment(Xtest, Ytest, idbn, run, plot_specs)
        
        mse_greedy[run] = np.array(mseg)
        mse_iterative[run] = np.array(msei)
    #end
    
    # plot the run-wise averages
    gcvis.plot_reconstruction_errors(mse_greedy, mse_iterative, plot_specs)
#end


def receptive_fields(plot_specs):
    '''
    Collect the model of interest (depending on the global specifications) and plot its receptive fields
    Input:
        - plot_specs (dict): see above
    Output:
        none
    '''
    
    dropt = '_drop' + str(plot_specs['dropout']).replace('.','d') if plot_specs['dropout'] < 1 else '_nodrop'
    gcorr = '' if plot_specs['gcorrection'] == 1 else '_' + str(plot_specs['gcorrection']).replace('.', 'd')
    
    for model in ['greedy','iterative']:
        print('Train {}'.format(model))
        mname = r'/series_{}runs_{}/{}_{}_dbn0_cd{}_ep{}{}{}.pkl'.format(plot_specs['runs'], 
                                                               plot_specs['scheme'],
                                                               plot_specs['dsid'], model, 
                                                               plot_specs['mcmc_steps'], 
                                                               plot_specs['epochs'], 
                                                               dropt,
                                                               gcorr
                                                              )
        with open(plot_specs['path_models'] + mname, 'rb') as f: 
            dbn = torch.load(f, map_location = 'cpu') 
        #end
        f.close()
        
        # the MNIST trained network has 4 layers of units,
        # hence the weight matrices products are ad-hoc
        W0 = dbn.rbm_layers[0].W
        W1 = torch.mm(dbn.rbm_layers[1].W, W0)
        if plot_specs['dsid']:
            W2 = torch.mm(dbn.rbm_layers[2].W, W1)
        
        gcvis.receptive_fields_save(W0, model, plot_specs, layer = '0')
        gcvis.receptive_fields_save(W1, model, plot_specs, layer = '1')
        if plot_specs['dsid']:
            gcvis.receptive_fields_save(W2, model, plot_specs, layer = '2')
    #end
#end


def plotPsychometricCurves(Xtrain, Ytrain, Xtest, Ytest, global_specs):
    '''
    The function that plots the psychometric curve given the activity values of 
    a given DBN layer. This is done by fitting the Delta Rule classifier on the 
    activities, so to get it to correctly disciminate whether a given pattern is 
    greater or smaller than a reference number, among 8 and 16.
    This is done on the SZ dataset.
    Input: 
        - X, Y (torch.Tensor): hidden layer activities and labels
        - global_specs (dict): global variables
    Output:
        - pfit[0] (float): estimated parameter of the curve, that is the
                           steepness of the sigmoidal curve. This value is 
                           the so-called Weber fraction
    '''
    
    import linearclassifier as bc
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit as cf
    
    psydata = {}
    
    for nref in global_specs['discrimination_Nref_ranges'].keys():
        psydata.update({ nref : { 'ratios' : [], 'percs'  : [] } })
        
        global_specs['web_nref'] = nref
        print('\nNumber ref: {}'.format(nref))
        
        dr = bc.DeltaRuleclassifier(Xtrain.shape[-1], global_specs['data_classes'], global_specs['web_nref'], 
                                    global_specs['lc_hyperparams'], global_specs['discrimination_Nref_ranges'])
        
        dr.train(Xtrain.clone(), Ytrain.clone(), global_specs)
        
        # the next line obtains the percentages of correct answers for any numerical ratio
        psydata[nref]['ratios'], psydata[nref]['percs'] = dr.getDiscriminationResults(Xtest.view(-1, Xtest.shape[-1]), Ytest.view(-1,1))
    #end
    
    def func(r, w):
        # definition of the sigmoid curve
        from scipy.stats import norm
        return norm.sf(0, loc = np.log(r), scale = np.sqrt(2) * w)
    #end
    
    # the ratios and percentages are gathered in two devoted data structures
    r8 = np.array(psydata[8]['ratios'])
    y8 = np.zeros(r8.size)
    for i in range(y8.size):
        y8[i] = psydata[8]['percs'][i].cpu().numpy()
    #end
    
    r16 = np.array(psydata[16]['ratios'])
    y16 = np.zeros(r16.size)
    for i in range(y16.size):
        y16[i] = psydata[16]['percs'][i].cpu().numpy()
    #end
    
    ra = np.sort(np.hstack((r8, r16)))
    ya = np.sort(np.hstack((y8, y16)))
    
    pfit, _ = cf(func, ra, ya, method = 'lm')
    
    fig, ax = plt.subplots(figsize = (4,3), dpi = 100)
    ax.plot(ra, func(ra, pfit), lw = 2, color = 'b', alpha = 0.75, label = 'Sigmoid fit')
    ax.scatter(r8, y8, color = 'k', marker = 'o', s = 20, alpha = 0.75, label = 'Nref = 8')
    ax.scatter(r16, y16, color = 'k', marker = '^', s = 20, alpha = 0.75, label = 'Nref = 16')
    title = '$w$ = {:.2f}'.format(pfit[0], global_specs['algorithm'])
    ax.set_title('Psychometric curve', fontsize = 18)
    # ax.text(1.25, 0.35, title, fontsize = 14)
    ax.set_xlabel('Numerical ratios', fontsize = 14)
    ax.set_ylabel('Accuracy', fontsize = 14)
    ax.legend(loc = 'lower right')
    fig.savefig(global_specs['path_images'] + r'/{}/profiles_and_errors/weberfrac_{}_cepoch_{}.pdf'.format(global_specs['scheme'], 
                                                global_specs['algorithm'], global_specs['current_epoch']), 
                                                format = 'pdf', bbox_inches = 'tight', dpi = 300)
    plt.show()
    
    print('Weber fraction estimate: {:.2f}'.format(pfit[0]))
    return pfit[0]
#end



def getNumerosityEstimate(Xtrain, Ytrain, Xtest, Ytest, global_specs):
    '''
    Obtain the results of the numerosity estimation task, given the
    activities Xtrain and Xtest.
    Input:
        - Xtrain, Ytrain, Xtest, Ytest (torch.Tensor): train and test sets
        - global_specs (dict): as above
    Output:
        - psydata (pandas.DataFrame): dataframe containing the results of the estimation
    '''
    
    import linearclassifier as lc
    
    mlp = lc.MLPclassifier(Xtrain.shape[-1], global_specs['data_classes'], global_specs)
    mlp.train(Xtrain, Ytrain, global_specs)
    psydata = mlp.numerosity_estimation(Xtest.clone(), Ytest.clone(), global_specs)
    
    return psydata
#end
    
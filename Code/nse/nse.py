
'''
Network science analysis of the DBN-reliant networks.
DBN have real-valued connection strenghts, hence the necessity to set a cut-off threshold
value. 
One can either mine the properties of the network for a given point (c, e) in the space defined
by all the values of cut-off threshold and epochs stamps, or repeat the analyses on a discrete
grid identified by suitably chosen coordinates {c_1, ..., c_N} x {e_1, ..., e_M}.
This latter strategy allows one to visualize things such as the mean degree, mean geodesic distance,
as functions of the training time and cut-off value.

Aim: for a given network, 1.) 1.1) Evaluate the degrees distribution for a given cut-off value
                              1.2) Visualize the parameters histograms
                              1.3) If one wishes to, visualize the eigenvalues of the adjacency matrix
                          2.) 2.1) Evaluate mean degree, mean geodesic distance and number of connected
                                   components as functions of the trainig time (epochs stamps) and the
                                   severeness of the preventive network binarization
                              2.2) Save and postprocess the results

``Any given network'' refers to the network identified by dataset, scheme, algorithm and dropout.
The DBN-reliant network is saved on disk from the previous segment of the work. Based on this network
model and its properties, the random replicas to compare the structure with, are generated.
'''

import os
import sys
sys.path.append(os.getcwd() + r'\..\dbp')

from pathlib import Path
import pandas as pd
import pickle

import gsd, gvc

'''
The user sets dataset specifier, initialization scheme, algorithm and dropout percentage
'''
dsid = 'mnist'
scheme = 'glorot'
alg = 'iterative'
dropout = '0d1'

if dsid == 'sz':
    final_training_epoch = 100
elif dsid == 'mnist':
    final_training_epoch = 50
#end

'''
RECALL: depending on the dataset, thr DBN architecture and hyperparams should be set accordingly.
Hence the sample epochs in which analyse the model change.
'''
if dsid == 'sz':
    init_schemes = ['normal']
    drop_percs = ['nodrop']
    if alg == 'greedy':
        epochs = [0, 5, 10, 20, 30, 50, 75, 90, 100, 105, 110, 120, 130, 150, 175, 190]
    elif alg == 'iterative':
        epochs = [0, 5, 10, 20, 30, 50, 75, 90]
    #end
elif dsid == 'mnist':
    init_schemes = ['normal', 'glorot']
    drop_percs = ['nodrop', '0d1']
    if alg == 'greedy':
        epochs = [5,20,40, 55,70,90, 105,120,140]
    elif alg == 'iterative':
        epochs = [3,5,10,15,20,25,30,35,40,49]
    #end
#end

'''
Control data structure to instruct the program
'''
flow   = {'sga'         : False,
          
          'savefig'     : True,
          'whist'       : False,
          'ddist'       : True,
          
          'dsid'        : dsid,
          'scheme'      : scheme,
          'alg'         : alg,
          'dropout'     : dropout,
          
          'distribution': 'p',
          
          'threshold'   : 0.4,
          'epoch'       : final_training_epoch,
          
           'epochs'     : epochs,
           'threlist'   : [0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5]
         }



if not flow['sga']:
    
    '''
    One single network for a given epoch (the last one typically) and a given cut-off value
    '''
    path_images = str(Path(os.getcwd()).parent) + r'\images\{}\networks\ep50'.format(flow['scheme'])
    flow['pathimages'] = path_images
    if not os.path.exists(path_images): os.system('mkdir ' + path_images)
    
    '''
    get the saved DBN network
    '''
    alg = 'greedy'
    wheremodel = r'\models\series_10runs_{}\{}_{}_dbn0_cd1_ep{}_{}.pkl'.format(flow['scheme'], dsid, alg, flow['epoch'], flow['dropout'])
    path_model = str(Path(os.getcwd()).parent) + wheremodel
    model_params = gsd.ModelStream.from_torch(path_model)
    ggraph = gsd.RealGraph(model_params, alg, flow['scheme'])
    
    alg = 'iterative'
    wheremodel = r'\models\series_10runs_{}\{}_{}_dbn0_cd1_ep{}_{}.pkl'.format(flow['scheme'], dsid, alg, flow['epoch'], flow['dropout'])
    path_model = str(Path(os.getcwd()).parent) + wheremodel
    model_params = gsd.ModelStream.from_torch(path_model)
    igraph = gsd.RealGraph(model_params, alg, flow['scheme'])
    
    if flow['whist']:
        whist = gvc.Histogram(flow)
        whist.plot([ggraph, igraph], nbins = 100, scale = {'x' : 'linear', 'y' : 'linear'})
    #end
    
    '''
    Crucial step! 
    Here the graphs are pruned. Refer to the graph data structures listed in the proper file, 
    according to whether a network is real or synthetic, pruning strategies may differ
    '''
    ggraph.prune(flow['threshold'])
    igraph.prune(flow['threshold'])
    
    if flow['whist']:
        whist.plot([ggraph, igraph], nbins = 100, scale = {'x' : 'log', 'y' : 'log'})
    #end
    
    '''
    Degrees distributions evaluation. The argument is the probability function to 
    evaluate, among these explained in the appendix B of Zambra, Testolin and Zorzi (in preparation)
    '''
    ggraph.degreesDistribution(p = flow['distribution'])
    igraph.degreesDistribution(p = flow['distribution'])
    
    ddist = gvc.DegreesDistribution(flow)     # why use plot function modules when you can design a CLASSES HIERARCHY!!! for plotting? fuck me
    for xyscale in ['log']:
        ddist.plot([ggraph, igraph], scale = {'x' : xyscale, 'y' : 'log'})
    #end
    
    
    '''
    Now, according to the real graphs structures and properties, similar random replicas are generated
    '''
    fggraph = gsd.Graph.getFakeGraph_like(ggraph, 'greedy', flow['scheme'], pscale = 1)
    figraph = gsd.Graph.getFakeGraph_like(igraph, 'iterative', flow['scheme'], pscale = 1)
    
    fggraph.prune()
    figraph.prune()
    fggraph.degreesDistribution(p = flow['distribution'])
    figraph.degreesDistribution(p = flow['distribution'])
    
    for xyscale in ['log']:
        ddist.plot([fggraph, figraph], titlename = 'alg', scale = {'x' : xyscale, 'y' : 'log'}, save = True)
    #end
    
else:
    
    '''
    May take ~8 hours of runtime
    
    Both the initialization schemes are accounted for, all the dropout percentages
    '''
    for scheme in init_schemes:
        
        for drop in drop_percs:
            
            flow['dropout'] = drop
            flow['scheme'] = scheme
            
            path_images = str(Path(os.getcwd()).parent) + r'\images\{}\networks\early'.format(flow['scheme'])
            flow['pathimages'] = path_images
            
            wheremodel = r'\models\epochs_tmp\{}_{}'.format(dsid, flow['scheme'])
            path_model = str(Path(os.getcwd()).parent) + wheremodel
            
            '''
            Initialize empty dataframes to store the results on the analyses in. 
            So that, once the mining stage has finished, one can easily fetch them
            and visualize the results
            '''
            flow['r_averages'] = {'k' : pd.DataFrame(columns = flow['threlist'],
                                                     index   = flow['epochs']),
                                  'd' : pd.DataFrame(columns = flow['threlist'],
                                                     index   = flow['epochs']),
                                  'ccs' : pd.DataFrame(columns = flow['threlist'],
                                                       index   = flow['epochs'])}
            
            flow['f_averages'] = {'k' : pd.DataFrame(columns = flow['threlist'],
                                                     index   = flow['epochs']),
                                  'd' : pd.DataFrame(columns = flow['threlist'],
                                                     index   = flow['epochs']),
                                  'ccs' : pd.DataFrame(columns = flow['threlist'],
                                                       index   = flow['epochs'])}
            
            '''
            For all the epochs stamps and all the threshold values
            '''
            for epoch in flow['epochs']:
                print('\n---Epoch {}'.format(epoch))
                
                for c in flow['threlist']:
                    
                    print('\nThreshold: {:.2f}'.format(c))
                    
                    flow['threshold'] = c
                    
                    params = gsd.ModelStream.from_torch(path_model + \
                             r'\{}_{}_dbn0_cd1_ep{}_{}.pkl'.format(dsid, flow['alg'], 
                                                                epoch, 
                                                                flow['dropout']))
                        
                    rgraph = gsd.RealGraph(params, flow['alg'], flow['scheme'])
                    
                    rgraph.prune(flow['threshold'])
                    rgraph.degreesDistribution(p = flow['distribution'])
                    flow['r_averages']['k'].loc[epoch][c] = rgraph.getAverageDegree();
                    flow['r_averages']['d'].loc[epoch][c] = rgraph.geoDistDistribution('mean')
                    flow['r_averages']['ccs'].loc[epoch][c] = rgraph.getEigs(get_ccs = True)
                    
                    fgraph = gsd.Graph.getFakeGraph_like(rgraph, flow['alg'], flow['scheme'], pscale = 1)
                    fgraph.prune()
                    fgraph.degreesDistribution(p = flow['distribution'])
                    flow['f_averages']['k'].loc[epoch][c] = fgraph.getAverageDegree()
                    flow['f_averages']['d'].loc[epoch][c] = fgraph.geoDistDistribution('mean')
                    flow['f_averages']['ccs'].loc[epoch][c] = fgraph.getEigs(get_ccs = True)
                    
                #end
            #end
            
            '''
            Save the results to disk
            '''
            with open('{}_{}_{}_{}_raves_{}.pkl'.format(dsid, flow['distribution'],
                                                     flow['alg'],
                                                     flow['scheme'],
                                                     flow['dropout']
                                                    ), 'wb') as f:
                pickle.dump(flow['r_averages'], f)
            #end
            f.close()
            
            with open('{}_{}_{}_{}_faves_{}.pkl'.format(dsid, flow['distribution'],
                                                     flow['alg'],
                                                     flow['scheme'],
                                                     flow['dropout']
                                                    ), 'wb') as f:
                pickle.dump(flow['f_averages'], f)
            #end
            f.close()
        #end
    #end
#end





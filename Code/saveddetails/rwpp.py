
import pandas as pd
import numpy as np
import pickle


''' Load and save '''
def load(path):
    
    with open(path, 'rb') as f:
        wdf = pickle.load(f)
    f.close(); #end
    return wdf
#end

def save(path, wdf):
    
    with open(path, 'wb') as f:
        pickle.dump(wdf, f)
    f.close(); #end
#end


''' WDF modifications '''
def insertion(wdf_to_be_innested, wdf_to_innest, position):
    
    wdf_to_be_innested.at[position] = wdf_to_innest.loc[position]
    return wdf_to_be_innested
#end

def join(wdf1, wdf2):
    
    wdf = dict()
    
    runs1 = list(wdf1.keys()).__len__()
    for i in range(runs1):
        wdf.update({i : wdf1[i]})
    #end
    
    runs2 = list(wdf2.keys()).__len__()
    for i in range(runs2):
        wdf.update({i + runs1 : wdf2[i]})
    #end
    
    return wdf
#end

def _join(wdfs):
    
    wdf = dict()
    ndfs = wdfs.__len__()
    runs_prev = 0
    
    for i in range(ndfs):
        
        runs = list(wdfs[i].keys()).__len__()
        
        for j in range(runs):
            wdf.update({i + j + runs_prev : wdfs[i][j]})
        #end
        
        runs_prev += (runs - 1)
    #end
    
    return wdf
#end

def convert_to_numeric(df):
    
    if type(df) == dict:
        
        for i in range(list(df).__len__()):
            for name in df[i].columns:
                
                df[i][name] = pd.to_numeric(df[i][name], errors = 'coerce')
            #end
        #end
    
    elif type(df) == pd.core.frame.DataFrame:
        
        for name in df.columns:
            df[name] = pd.to_numeric(df[name], errors = 'coerce')
        #end
    #end
    
    return df
#end


def outlier_remove(wdf, threshold = 10):
    
    for i in range(list(wdf.keys()).__len__()):
        
        if np.any(wdf[i].values >= threshold) or np.any(wdf[i].values <= -threshold):
            
            for j in wdf[i].columns.tolist():
                if np.any(wdf[i][j].values >= threshold) or np.any(wdf[i][j].values <= -threshold):
                    wdf[i] = wdf[i].drop(columns = [j])
                #end
            #end
        #end
    #end
    
    return wdf
#end


def clean(wdf, scale_stds = 1):
    
    for i in range(list(wdf.keys()).__len__()):
        
        W = wdf[i].values
        
        W_mean = W.mean(axis = 1)
        W_stds = W.std(axis = 1)
        
        indices_to_delete = list()
        columns_dataframe = np.array(wdf[i].columns, dtype = np.int64)
        for j in range(W.shape[1]):
            if np.sum(W[:,j] < W_mean - scale_stds * W_stds) + np.sum(W[:,j] > W_mean + scale_stds * W_stds) >= 1:
                indices_to_delete.append(j)
            #end
        #end
        
        wdf[i] = wdf[i].drop(columns = columns_dataframe[indices_to_delete])
        if indices_to_delete.__len__() == W.shape[1]:
            del wdf[i]
        #end
    #end
    
    _wdf = dict()
    for i, key in enumerate(wdf):
        _wdf.update({i : wdf[key]})
    #end
    
    return _wdf
#end


def remove(wdf, run, classifier):
    
    wdf[run] = wdf[run].drop(columns = [classifier])
    return wdf
#end

def scale(wdf, scale):
    
    if scale == 'SZ':  scalefact = np.sqrt(1)
    if scale == 'TZM': scalefact = np.sqrt(2)
    
    for i in range(list(wdf.keys()).__len__()):
        wdf[i] = wdf[i] * scalefact
    #end
    
    return wdf
#end


''' Details '''
def get_numruns(wdf):
    return list(wdf.keys()).__len__()
#end

def get_numlcs(wdf):
    runs = get_numruns(wdf)
    numlcs = 0
    for i in range(runs):
        numlcs += wdf[i].shape[1]
    #end
    return numlcs
#end

def all_dfs_have_the_same_epochs(wdf):
    runs = get_numruns(wdf)
    positive_outcome = True
    for i in range(1, runs):
        if not wdf[i-1].shape[0] == wdf[i].shape[0]:
            print('Number of reference epochs mismatch')
            positive_outcome = False
        if not np.all(np.array(wdf[i-1].index) == np.array(wdf[i].index)):
            print('Reference epochs stamps mismatch')
            positive_outcome = False
        #end
    #end
    return positive_outcome
#end

def get_numepochs(wdf):
    if all_dfs_have_the_same_epochs(wdf):
        return wdf[0].shape[0]
    else:
        raise IndexError
    #end
#end

def get_sampleepochs(wdf):
    if all_dfs_have_the_same_epochs(wdf):
        return np.array(wdf[0].index)
    else:
        raise IndexError
    #end
#end




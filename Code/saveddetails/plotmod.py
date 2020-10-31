
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit as cf



def preprocess(wdf, scale, runs):
    
    for i in range(runs):
        for name in wdf[i].columns:
            wdf[i][name] = pd.to_numeric(wdf[i][name], errors = 'coerce')
        #end
        # wdf[i] = wdf[i] * scalefact
    #end
    reference_epochs = wdf[0].shape[0]
    
    
    W = np.zeros((reference_epochs, runs))
    for i in range(runs):
        W[:,i] = wdf[i].mean(axis = 1)
    #end
    
    sample_epochs_index = np.array(wdf[0].index)
    sparse_data_points = pd.DataFrame(columns = ['mean_w', 'std_w'], index = np.arange(0, sample_epochs_index[-1]+1))
    
    sparse_data_points['mean_w'].loc[sample_epochs_index] = W.mean(axis = 1)
    sparse_data_points['std_w'].loc[sample_epochs_index] = W.std(axis = 1)
    
    for name in sparse_data_points.columns:
        sparse_data_points[name] = pd.to_numeric(sparse_data_points[name], errors = 'coerce')
    #end
    sparse_data_averages = np.array(sparse_data_points['mean_w'].values)
    sparse_data_std = np.array(sparse_data_points['std_w'].values)
    
    ''' related to all the learning time '''
    all_index = np.arange(0, sparse_data_averages.size)
    
    sparse_data_averages_nonan = sparse_data_averages[~np.isnan(sparse_data_averages)]
    sparse_data_std_nonan = sparse_data_std[~np.isnan(sparse_data_std)]
    
    return W, sample_epochs_index, all_index, sparse_data_averages_nonan, sparse_data_std_nonan
#end




def getTitle(pfit, scale, tag):
    if tag == 'simple':
        text = '$y = {:.2f}\,x^{{{:.2f}}}$'.format(pfit[0], pfit[1])
    elif tag == 'CAB':
        text = '$y = {:.2f}\,(1 + \,x)^{{{:.2f}}}$'.format(pfit[0], pfit[1])
    elif tag == 'CABS':
        text = '$y = {:.2f}\,(1 + {:.2f}\,x)^{{{:.2f}}}$'.format(pfit[0], pfit[1], pfit[2])
    #end
    if scale['scaleid'] == 'SZ':
        pos = (40, 0.3)
    elif scale['scaleid'] == 'TZM':
        pos = (40, 0.4)
    #end
    return text, pos
#end


def plotWeberFracTrend(W, wdf, x_all, x_sample, sparse_data_averages, sparse_data_std, scale):
    run_wise_mean = W.mean(axis = 1)
    run_wise_std = W.std(axis = 1)
    span = np.arange(0, run_wise_mean.size)
    xticks = np.array(wdf[0].index)
    
    run_wise_mean = sparse_data_averages
    run_wise_std  = sparse_data_std
    
    fig, ax = plt.subplots(figsize = (4,3), dpi = 150)
    ax.plot(x_sample, run_wise_mean, color = 'b', lw = 2, alpha = 0.85, label  = 'DeltaRule-wise average')
    ax.fill_between(x_sample, run_wise_mean, run_wise_mean + run_wise_std, color = 'b', alpha = 0.15, label = 'Runs-wise average')
    ax.fill_between(x_sample, run_wise_mean, run_wise_mean - run_wise_std, color = 'b', alpha = 0.15)
    ax.set_xlabel('Reference Epoch', fontsize = 14)
    # ax.set_xticks(np.arange(0, xticks.size))
    # ax.set_xticklabels(xticks)
    y = x_all[x_all % 10 == 0]
    ax.set_xticks(y)
    ax.set_xticklabels(y)
    ax.set_ylabel('Weber Fraction', fontsize = 14)
    ax.set_title('Weber fraction trend', fontsize = 16)
    # ax.legend()
    fig.savefig(os.getcwd() + r'\images\weberfrac_trend_{}.pdf'.format(scale['scaleid']), format = 'pdf', bbox_inches = 'tight', dpi = 300)
    plt.show(fig)
#end


def scatterplot(x_all, y_all, x_sample, y_sample, y_sample_err, pfit, scale, tag = '', rsquare = ''):
    titledict = {'simple' : 'Simple Power-law',
                 'CAB'    : 'Competence at Birth Power-law',
                 'CABS'   : 'Competence at Birth (scaled) Power-law'} 
    
    fig, ax = plt.subplots(figsize = (4,3), dpi = 150)
    ax.plot(x_all, y_all, color = 'k', lw = 2, alpha = 0.75, label = 'Fit: $R^2$ = {:.2f}'.format(rsquare))
    if scale['scaleid'] == 'SZ':
        ax.plot(x_all, np.ones(x_all.size) * 0.15, lw = 1, ls = '--', color = 'r', alpha = 0.75, label = 'S&Z')
    if scale['scaleid'] == 'TZM':
        ax.plot(x_all, np.ones(x_all.size) * 0.22, lw = 1, ls = '--', color = 'g', alpha = 0.75, label = 'TZM')
    ax.scatter(x_sample, y_sample, marker = '_', s = 10, alpha = 0.75, color = 'b', label = 'Sample points')
    ax.errorbar(x_sample, y_sample, yerr = y_sample_err, fmt = 'none', color = 'b', alpha = 0.75)
    ax.set_xlabel('Epoch', fontsize = 14)
    y = x_all[x_all % 10 == 0]
    ax.set_xticks(y)
    ax.set_xticklabels(y)
    ax.set_ylabel('Weber Fraction', fontsize = 14)
    # ax.set_title(titledict[tag], fontsize = 16)
    ax.set_title('Weber fraction development', fontsize = 14)
    text, pos = getTitle(pfit, scale, tag); #ax.text(pos[0], pos[1], text, fontsize = 14, alpha = 0.75)
    # ax.legend()
    fig.savefig(os.getcwd() + r'\images\weberfrac_{}Powerlaw_fit__{}.pdf'.format(tag, scale['scaleid']), format = 'pdf', bbox_inches = 'tight', dpi = 300)
    plt.show()
    
    with open(os.getcwd() + r'/FitReport_{}_{}.txt'.format(tag, scale['scaleid']), 'w') as f:
        f.write('Parameters fit, in equation form:\n')
        f.write(text)
        f.write('\nR^2 = {:.2f}'.format(rsquare))
    #end
    f.close()
#end
    


def fit_simple_powerlaw(sample_epochs_index, all_index, sparse_data_averages_nonan, sparse_data_std_nonan, scale):
    def plfunc(x, a, b):
        return a * np.power(x, -b)
    #end
    
    pfit, _ = cf(plfunc, sample_epochs_index[1:], sparse_data_averages_nonan[1:])
    residuals = sparse_data_averages_nonan[1:] - plfunc(sample_epochs_index[1:], *pfit)
    squared_res = np.sum(np.power(residuals, 2))
    sum_tot_squares = np.sum( (sparse_data_averages_nonan - np.mean(sparse_data_averages_nonan))**2 )
    r_squared = 1 - (squared_res / sum_tot_squares)
    
    all_data = plfunc(all_index[1:], *pfit)
    
    scatterplot(all_index[1:], all_data, sample_epochs_index, sparse_data_averages_nonan, sparse_data_std_nonan, pfit, scale, tag = 'simple', rsquare = r_squared)
    
    return pfit
#end


def fit_CAB_powerlaw(sample_epochs_index, all_index, sparse_data_averages_nonan, sparse_data_std_nonan, scale):
    def plfunc(x, a, b):
        return a * np.power(1 + x, -b)
    #end

    pfit, _ = cf(plfunc, sample_epochs_index, sparse_data_averages_nonan)
    residuals = sparse_data_averages_nonan - plfunc(sample_epochs_index, *pfit)
    squared_res = np.sum(np.power(residuals, 2))
    sum_tot_squares = np.sum( (sparse_data_averages_nonan - np.mean(sparse_data_averages_nonan))**2 )
    r_squared = 1 - (squared_res / sum_tot_squares)
    
    all_data = plfunc(all_index, *pfit)
    scatterplot(all_index, all_data, sample_epochs_index, sparse_data_averages_nonan, sparse_data_std_nonan, pfit, scale, tag = 'CAB', rsquare = r_squared)
    
    return pfit
#end


def fit_CABS_powerlaw(sample_epochs_index, all_index, sparse_data_averages_nonan, sparse_data_std_nonan, scale):
    def plfunc(x, a, s, b):
        x_ = np.ones(x.size) + s * x 
        return a * np.power(x_, -b)
    #end
    import warnings
    warnings.simplefilter("ignore")
    
    pfit, _ = cf(plfunc, sample_epochs_index, sparse_data_averages_nonan, 
                 p0 = [1., 1., 1.], maxfev = 2000)
    residuals = sparse_data_averages_nonan - plfunc(sample_epochs_index, *pfit)
    squared_res = np.sum(np.power(residuals, 2))
    sum_tot_squares = np.sum( (sparse_data_averages_nonan - np.mean(sparse_data_averages_nonan))**2 )
    r_squared = 1 - (squared_res / sum_tot_squares)
    
    all_data= plfunc(all_index, *pfit)
    scatterplot(all_index, all_data, sample_epochs_index, sparse_data_averages_nonan, sparse_data_std_nonan, pfit, scale, tag = 'CABS', rsquare = r_squared)
    
    return pfit
#end
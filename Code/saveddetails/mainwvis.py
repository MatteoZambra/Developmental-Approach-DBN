import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
import rwpp as wv
import os
import sys
import argparse

from plotmod import plotWeberFracTrend, fit_simple_powerlaw, fit_CAB_powerlaw, fit_CABS_powerlaw



parser = argparse.ArgumentParser(description = 'Arguments -- Weber fraction trend and powerlaw fit')
parser.add_argument('-plttr', type = str, help = 'ptrd: t (True) or f (False), whether to plot the Weber fraction trend during learning')
parser.add_argument('-plfit', type = str, help = 'plft: t (True) or f (False), whether to plot the power law fit of Weber fraction trend')
parser.add_argument('-plfun', type = str, help = 'plfc: among _simple_, _CAB_, _CABS_, respectively, simple powerlaw: y = a * x^b; competence at birth: y = a * (1 + x)^b; competence at birth with scaling: y = a * (1 + s * x)^b. See Testolin, Zhou, McClelland (2020)')
parser.add_argument('-scale', type = str, help = 'scale: among SZ and TZM, respectively see Stoianov and Zorzi (2012) and Testolin et al (2020)')
args = parser.parse_args()


if sys.argv.__len__() == 1:
    plot_wtrend = 't'
    plot_wpowerlaw = 't'
    powerlaw_id = 'CABS'
    scale = 'TZM'
else:
    plot_wtrend = args.plttr
    plot_wpowerlaw = args.plfit
    powerlaw_id = args.plfun
    scale = args.scale
#end

powerlaw_toolbox = {'simple' : fit_simple_powerlaw,
                    'CAB'    : fit_CAB_powerlaw,
                    'CABS'   : fit_CABS_powerlaw}


wdf = wv.load(os.getcwd() + r'\raw_wdata\wdf_preprocss.pkl')
wdf = wv.scale(wdf, scale = scale)
runs = list(wdf.keys()).__len__()
reference_epochs = wdf[0].shape[0]

W = np.zeros((reference_epochs, runs))
for i in range(runs):
    W[:,i] = wdf[i].mean(axis = 1)
#end

''' only related to sample points '''
sample_epochs_index = np.array(wdf[0].index)
sparse_data_points = pd.DataFrame(columns = ['mean_w', 'std_w'], index = np.arange(0, sample_epochs_index[-1]+1))

sparse_data_points['mean_w'].loc[sample_epochs_index] = W.mean(axis = 1)
sparse_data_points['std_w'].loc[sample_epochs_index] = W.std(axis = 1)
sparse_data_points = wv.convert_to_numeric(sparse_data_points)

sparse_data_averages = np.array(sparse_data_points['mean_w'].values)
sparse_data_std = np.array(sparse_data_points['std_w'].values)

all_index = np.arange(0, sparse_data_averages.size)

sparse_data_averages_nonan = sparse_data_averages[~np.isnan(sparse_data_averages)]
sparse_data_std_nonan = sparse_data_std[~np.isnan(sparse_data_std)]

if plot_wtrend == 't':
    plotWeberFracTrend(W, wdf, all_index, sample_epochs_index, 
                       sparse_data_averages_nonan, sparse_data_std_nonan, 
                       {'scaleid' : scale})
#end


if plot_wpowerlaw == 't':
    pfit = powerlaw_toolbox[powerlaw_id](sample_epochs_index, all_index, 
                                         sparse_data_averages_nonan,
                                         sparse_data_std_nonan, 
                                         {'scaleid' : scale})
#end

plt.close('all')


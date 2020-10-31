

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
import os
import rwpp as wv


deltarule_train = 'mini'

wdf = wv.load(os.getcwd() + r'\raw_wdata\wdf_preprocss_{}.pkl'.format(deltarule_train))

samples       =  wdf[0].shape[0]
numlcs        =  wdf[0].columns.__len__()
sample_epochs =  np.array(wdf[0].index)
numruns       =  wv.get_numruns(wdf)
numlcs        =  wv.get_numlcs(wdf)
samples       =  wv.get_numepochs(wdf)
sample_epochs =  wv.get_sampleepochs(wdf)

W = np.zeros((samples, numlcs))
colprev = 0
for i in range(numruns):
    numlcs = wdf[i].shape[1]
    colstart = colprev
    colend = colstart + numlcs
    W[:, colstart : colend] = wdf[i].values
    colprev = colend
#end

span = np.arange(0, W.shape[0])
averages = W.mean(axis = 1)
stds = W.std(axis = 1)


fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
ax.plot(span, W[:,0], color = 'b', lw = 1, alpha = 0.2, label = 'Trajectory')
ax.plot(span, W[:,1:], color = 'b', lw = 1, alpha = 0.2)
ax.scatter(span, averages, s = 10, color = 'k', alpha = 1.0, label = 'Mean')
ax.errorbar(span, averages, yerr = stds, fmt = 'none', color = 'k', label = 'StdDev')
ax.set_xlabel('Epochs', fontsize = 14)
ax.set_ylabel('Weber Fraction', fontsize = 14)
ax.set_xticks(np.arange(0, samples))
ax.set_xticklabels(sample_epochs)
ax.legend()
fig.savefig(os.getcwd() + r'\images\wtrajectories.png', format = 'png', bbox_inches = 'tight', dpi = 300)
plt.show()

Wminima = np.zeros((samples, numruns))
Wmaxima = np.zeros((samples, numruns))
run_min = np.zeros(samples)
run_max = np.zeros(samples)

for i in range(numruns):
    Wminima[:,i] = wdf[i].min(axis = 1)
    Wmaxima[:,i] = wdf[i].max(axis = 1)
    run_min = W.min(axis = 1)
    run_max = W.max(axis = 1)
#end

fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
span = np.arange(0, W.shape[0])
ax.plot(span, Wminima[:,0], color = 'b', lw = 1, alpha = 0.5, label = 'Minima')
ax.plot(span, Wminima[:,1:], color = 'b', lw = 1, alpha = 0.5)
ax.plot(span, Wmaxima[:,0], color = 'r', lw = 1, alpha = 0.5, label = 'Maxima')
ax.plot(span, Wmaxima[:,1:], color = 'r', lw = 1, alpha = 0.5)
ax.scatter(span, averages, s = 10, color = 'k', alpha = 1.0, label = 'Mean')
ax.errorbar(span, averages, yerr = stds, fmt = 'none', color = 'k', label = 'StdDev')
ax.set_xlabel('Epochs', fontsize = 14)
ax.set_ylabel('Weber Fraction', fontsize = 14)
ax.set_xticks(np.arange(0, samples))
ax.set_xticklabels(sample_epochs)
ax.legend()
fig.savefig(os.getcwd() + r'\images\wtrajectories_minmax.png', format = 'png', bbox_inches = 'tight', dpi = 300)
plt.show()

min_mean = Wminima.mean(axis = 1)
min_lowerbound = Wminima.min(axis = 1)
min_upperbound = Wminima.max(axis = 1)
max_mean = Wmaxima.mean(axis = 1)
max_lowerbound = Wmaxima.min(axis = 1)
max_upperbound = Wmaxima.max(axis = 1)

fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
span = np.arange(0, W.shape[0])
ax.plot(span, min_mean, color = 'b', ls = '--', lw = 1, alpha = 0.75, label = 'Minima Mean')
ax.fill_between(span, min_lowerbound, min_upperbound, color = 'b', alpha = 0.25)
ax.plot(span, max_mean, color = 'r', ls = '--', lw = 1, alpha = 0.75, label = 'Maxima Mean')
ax.fill_between(span, max_lowerbound, max_upperbound, color = 'r', alpha = 0.25)
ax.scatter(span, averages, s = 10, color = 'k', alpha = 1.0, label = 'Mean')
ax.errorbar(span, averages, yerr = stds, fmt = 'none', color = 'k', label = 'StdDev')
ax.plot(span, np.ones(span.size) * 0.15, lw = 1, alpha = 0.5, color = 'k')
ax.set_xlabel('Epochs', fontsize = 14)
ax.set_ylabel('Weber Fraction', fontsize = 14)
ax.set_xticks(np.arange(0, samples))
ax.set_xticklabels(sample_epochs)
ax.legend()
fig.savefig(os.getcwd() + r'\images\wtrajectories_mmeans.png', format = 'png', bbox_inches = 'tight', dpi = 300)
plt.show()
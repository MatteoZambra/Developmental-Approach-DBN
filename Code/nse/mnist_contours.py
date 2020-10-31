
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import os


def readable_format(n):
    if type(n) == str:
        n = float(n)
    #end
    _n = float('{:.3g}'.format(n))
    order_of_magnitude = 0
    while abs(_n) >= 1000:
        order_of_magnitude += 1
        _n /= 1000.0
    #end
    return '{}{}'.format('{:f}'.format(_n).rstrip('0').rstrip('.'), ['', 'K', 'M', 'G', 'T'][order_of_magnitude])
#end




scheme = 'normal'
drop   = 'nodrop'


greedy_epochs = [5,20,40, 55,70,90, 105,120,140]
greedy_xticks = greedy_epochs
greedy_xtickslabels = [5,20,40, 5,20,40, 5,20,40]
epoch_vline = [50, 100]

iterative_epochs = [3,5,10,15,20,25,30,35,40,49]
iterative_xticks = [10,20,30,40]
iterative_figwidth  = 4
iterative_figheight = 3
# yticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
yticks = [0.2, 0.6, 1.0, 1.5]

with open('datamaps\mnist\mnist_p_greedy_{}_raves_{}.pkl'.format(scheme, drop), 'rb') as f:
    greedy_raverages = pickle.load(f)
#end
f.close()
with open('datamaps\mnist\mnist_p_iterative_{}_raves_{}.pkl'.format(scheme, drop), 'rb') as f:
    iterative_raverages = pickle.load(f)
#end
f.close()
with open('datamaps\mnist\mnist_p_greedy_{}_faves_{}.pkl'.format(scheme, drop), 'rb') as f:
    greedy_faverages = pickle.load(f)
#end
f.close()
with open('datamaps\mnist\mnist_p_iterative_{}_faves_{}.pkl'.format(scheme, drop), 'rb') as f:
    iterative_faverages = pickle.load(f)
#end
f.close()

gravesk = greedy_raverages['k']
gravesd = greedy_raverages['d']
gravesc = greedy_raverages['ccs']
gfavesk = greedy_faverages['k']
gfavesd = greedy_faverages['d']
gfavesc = greedy_faverages['ccs']

iravesk = iterative_raverages['k']
iravesd = iterative_raverages['d']
iravesc = iterative_raverages['ccs']
ifavesk = iterative_faverages['k']
ifavesd = iterative_faverages['d']
ifavesc = iterative_faverages['ccs']

gc = np.array(gravesk.columns)
ge = np.array(gravesk.index)

ic = np.array(iravesk.columns)
ie = np.array(iravesk.index)

gc, ge = np.meshgrid(gc, ge)
ic, ie = np.meshgrid(ic, ie)

grk = gravesk.values
grd = gravesd.values
grc = gravesc.values
gfk = gfavesk.values
gfd = gfavesd.values
gfc = gfavesc.values

irk = iravesk.values
ird = iravesd.values
irc = iravesc.values
ifk = ifavesk.values
ifd = ifavesd.values
ifc = ifavesc.values

grk = np.array(grk, dtype = np.float64)
grd = np.array(grd, dtype = np.float64)
grc = np.array(grc, dtype = np.float64)
gfk = np.array(gfk, dtype = np.float64)
gfd = np.array(gfd, dtype = np.float64)
gfc = np.array(gfc, dtype = np.float64)

irk = np.array(irk, dtype = np.float64)
ird = np.array(ird, dtype = np.float64)
irc = np.array(irc, dtype = np.float64)
ifk = np.array(ifk, dtype = np.float64)
ifd = np.array(ifd, dtype = np.float64)
ifc = np.array(ifc, dtype = np.float64)

lw = 1
alpha = 1
fstitle = 16
fslegend = 8
fsticks = 12
fslabel = 12
cmapkd =  'gnuplot' #'gist_stern' #'nipy_spectral' #gnuplot #Reds, #Oranges
cmapccs = 'gnuplot'


fig = plt.figure(figsize = (10,6))
gs = gridspec.GridSpec(3,2, width_ratios = [1,2.65])
    
# REAL
# ITERATIVE real net, mean degree
ax_irk = plt.subplot(gs[0,0])

if irk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(irk.min(), irk.max(), increase)
cplot = ax_irk.contourf(ie, ic, irk, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_irk.set_title('Mean degree', fontsize = fstitle, pad = 10)
ax_irk.set_ylabel('Threshold', fontsize = fslabel)
ax_irk.tick_params(axis = 'both', labelsize = fsticks)
ax_irk.set_xticks(iterative_xticks)
ax_irk.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_irk)
cbar.ax.locator_params(nbins = 6)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)


# GREEDY real net, mean degree
ax_grk = plt.subplot(gs[0,1])
if grk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(grk.min(), grk.max(), increase)
cplot = ax_grk.contourf(ge, gc, grk, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_grk.set_title('Mean degree', fontsize = fstitle, pad = 10)
ax_grk.tick_params(axis = 'both', labelsize = fsticks)
ax_grk.set_xticks(greedy_xticks)
ax_grk.set_xticklabels(greedy_xtickslabels)
ax_grk.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_grk, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for epochvline in epoch_vline:
    ax_grk.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end

# ITERATIVE real net, mean distance
ax_ird = plt.subplot(gs[1,0])

if ird.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(ird.min(), ird.max(), increase)
cplot = ax_ird.contourf(ie, ic, ird, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_ird.set_title('Mean distance', fontsize = fstitle, pad = 10)
ax_ird.set_ylabel('Threshold', fontsize = fslabel)
ax_ird.tick_params(axis = 'both', labelsize = fsticks)
ax_ird.set_xticks(iterative_xticks)
ax_ird.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_ird)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)


# GREEDY real net, mean distance
ax_grd = plt.subplot(gs[1,1])
if grk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(grd.min(), grd.max(), increase)
cplot = ax_grd.contourf(ge, gc, grd, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_grd.set_title('Mean distance', fontsize = fstitle, pad = 10)
ax_grd.tick_params(axis = 'both', labelsize = fsticks)
ax_grd.set_xticks(greedy_xticks)
ax_grd.set_xticklabels(greedy_xtickslabels)
ax_grd.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_grd, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for epochvline in epoch_vline:
    ax_grd.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end


# ITERATIVE real net, components
ax_irc = plt.subplot(gs[2,0])

if irc.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
cplot = ax_irc.contourf(ie, ic, irc, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_irc.set_title('Components', fontsize = fstitle, pad = 10)
ax_irc.set_ylabel('Threshold', fontsize = fslabel)
ax_irc.set_xlabel('Epoch', fontsize = fslabel)
ax_irc.tick_params(axis = 'both', labelsize = fsticks)
ax_irc.set_xticks(iterative_xticks)
ax_irc.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_irc)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for tk in cbar.ax.yaxis.get_ticklabels():
    number = float(tk.get_text())
    tk.set_text(readable_format(number))
#end

# GREEDY real net, components
ax_grc = plt.subplot(gs[2,1])
if grc.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
cplot = ax_grc.contourf(ge, gc, grc, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_grc.set_title('Components', fontsize = fstitle, pad = 10)
ax_grc.set_xlabel('Epoch', fontsize = fslabel)
ax_grc.tick_params(axis = 'both', labelsize = fsticks)
ax_grc.set_xticks(greedy_xticks)
ax_grc.set_xticklabels(greedy_xtickslabels)
ax_grc.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_grc, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
# labels = []
# for tk in cbar.ax.yaxis.get_ticklabels():
#     number = float(tk.get_text())
#     # tk.set_text(readable_format(number))
#     labels.append(readable_format(number))
# #end
# cbar.ax.set_yticklabels(labels)
for epochvline in epoch_vline:
    ax_grc.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end

fig.tight_layout()
fig.savefig(os.getcwd() + r'\datamaps\images\mnist_kdc_{}_real_{}.pdf'.format(scheme, drop), format = 'pdf', dpi = 300, bbox_inches = 'tight')
plt.show(fig)




# FAKE ------------------------------------------------------------------------

fig = plt.figure(figsize = (10,6))
gs = gridspec.GridSpec(3,2, width_ratios = [1,2.65])

# ITERATIVE fake net, mean degree
ax_ifk = plt.subplot(gs[0,0])

if ifk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(ifk.min(), ifk.max(), increase)
cplot = ax_ifk.contourf(ie, ic, ifk, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_ifk.set_title('Mean degree', fontsize = fstitle, pad = 10)
ax_ifk.set_ylabel('Threshold', fontsize = fslabel)
ax_ifk.tick_params(axis = 'both', labelsize = fsticks)
ax_ifk.set_xticks(iterative_xticks)
ax_ifk.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_ifk)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)


# GREEDY fake net, mean degree
ax_gfk = plt.subplot(gs[0,1])
if gfk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(gfk.min(), gfk.max(), increase)
cplot = ax_gfk.contourf(ge, gc, gfk, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_gfk.set_title('Mean degree', fontsize = fstitle, pad = 10)
# ax_grk.set_ylabel('Threshold', fontsize = fslabel)
ax_gfk.tick_params(axis = 'both', labelsize = fsticks)
ax_gfk.set_xticks(greedy_xticks)
ax_gfk.set_xticklabels(greedy_xtickslabels)
ax_gfk.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_gfk, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for epochvline in epoch_vline:
    ax_gfk.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end

# ITERATIVE fake net, mean distance
ax_ifd = plt.subplot(gs[1,0])

if ifd.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(ifd.min(), ifd.max(), increase)
cplot = ax_ifd.contourf(ie, ic, ifd, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_ifd.set_title('Mean distance', fontsize = fstitle, pad = 10)
ax_ifd.set_ylabel('Threshold', fontsize = fslabel)
ax_ifd.tick_params(axis = 'both', labelsize = fsticks)
ax_ifd.set_xticks(iterative_xticks)
ax_ifd.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_ifd)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)


# GREEDY fake net, mean distance
ax_gfd = plt.subplot(gs[1,1])
if gfk.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
clev = np.arange(gfd.min(), gfd.max(), increase)
cplot = ax_gfd.contourf(ge, gc, gfd, clev, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_gfd.set_title('Mean distance', fontsize = fstitle, pad = 10)
ax_gfd.tick_params(axis = 'both', labelsize = fsticks)
ax_gfd.set_xticks(greedy_xticks)
ax_gfd.set_xticklabels(greedy_xtickslabels)
ax_gfd.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_gfd, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for epochvline in epoch_vline:
    ax_gfd.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end


# ITERATIVE fake net, components
ax_ifc = plt.subplot(gs[2,0])

if ifc.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
cplot = ax_ifc.contourf(ie, ic, ifc, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_ifc.set_title('Components', fontsize = fstitle, pad = 10)
ax_ifc.set_ylabel('Threshold', fontsize = fslabel)
ax_ifc.set_xlabel('Epoch', fontsize = fslabel)
ax_ifc.tick_params(axis = 'both', labelsize = fsticks)
ax_ifc.set_xticks(iterative_xticks)
ax_ifc.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_ifc)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)


# GREEDY fake net, components
ax_gfc = plt.subplot(gs[2,1])
if gfc.max() < 1:
    increase = 0.1
else:
    increase = 1
#end
cplot = ax_gfc.contourf(ge, gc, gfc, alpha = alpha, cmap = cmapkd)
for cf in cplot.collections:
    cf.set_edgecolor("face")
#end
ax_gfc.set_title('Components', fontsize = fstitle, pad = 10)
ax_gfc.set_xlabel('Epoch', fontsize = fslabel)
ax_gfc.tick_params(axis = 'both', labelsize = fsticks)
ax_gfc.set_xticks(greedy_xticks)
ax_gfc.set_xticklabels(greedy_xtickslabels)
ax_gfc.set_yticks(yticks)
cbar = fig.colorbar(cplot, ax = ax_gfc, pad = 0.015)
# tkls = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(tkls, fontsize = fslegend)
cbar.ax.locator_params(nbins = 6)
for epochvline in epoch_vline:
    ax_gfc.vlines(epochvline, 0.2, 1.5, color = 'w', lw = 3, alpha = 0.75, linestyle = 'solid')
#end

fig.tight_layout()
fig.savefig(os.getcwd() + r'\datamaps\images\mnist_kdc_{}_fake_{}.pdf'.format(scheme, drop), format = 'pdf', dpi = 300, bbox_inches = 'tight')
plt.show(fig)









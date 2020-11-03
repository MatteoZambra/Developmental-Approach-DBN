import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use('seaborn-ticks')




def receptive_fields_explore(W, model, plot_specs, layer):
    
    W = W.clone().cpu()
    Wmax = W.max(); Wmin = W.min()
    W = 2 * (W - Wmin)/(Wmax - Wmin) - 1
    hidden_dim = int(np.sqrt(W.shape[1]))
    
    for i in range(5):
        fig = plt.figure(figsize = (10,10))
        xdim = 10; ydim = 10
        offset = int(xdim * ydim)
        indices = list(range(xdim * ydim))
        for j in indices:
            ax = fig.add_subplot(xdim, ydim, j+1, xticks = [], yticks = [])
            index = i * offset + j
            ax.imshow(W[index,:].view(hidden_dim, hidden_dim), cmap = 'gray')
            plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
            ax.text(1, 1, str(int(index)), fontsize = 14, color = 'lime')
        #end
        fig.savefig(r'C:\users\matte\Desktop\rfs\{}_receptive_fields_{}.png'.format(model, i), format = 'png', dpi = 300, bbox_inches = 'tight')
        plt.show()
    #end
    
#end

def receptive_fields_save(W, model, plot_specs, layer):
    """
    Plot receptive fields, that is the combination of weights matrices.
    This is done to visualize what are the features that are learned by the 
    device. Some resemblances with the actual digits could be recognisable.
    Panels plotted are arbitrary, neurons from which these come are chosen randomly
    
    Input:
        ~ W (torch.Tensor) : weights matrix
        
    Returns:
        ~ nothing
    """
    
    
    W = W.clone().cpu()
    
    Wmax = W.max(); Wmin = W.min()
    W = 2 * (W - Wmin)/(Wmax - Wmin) - 1
    
    if layer == '2':
        if model == 'greedy':
            indices = [15, 711, 1027, 1109, 1306, 1415, 1403, 1549, 1152, 522, 1, 1298,
                       16, 11, 59, 120, 162, 260, 305, 425, 461, 565, 689, 618, 697]
        elif model == 'iterative':
            indices = [9, 107, 252, 267, 253, 535, 636, 688, 1802, 208, 747, 1552,
                       12, 263, 309, 442, 715, 883, 887, 952, 902, 1392, 1474, 1788, 1887]
        #end
    #end
    
    if layer == '1':
        if model == 'greedy':
            indices = [9, 12, 24, 26, 29, 36, 51, 52, 65, 111, 150, 165, 167, 218, 233,
                       228, 246, 299, 309, 330, 403, 426, 499, 438, 486]
        elif model == 'iterative':
            indices = [0, 3, 6, 43, 48, 83, 93, 95, 72, 61, 147, 171, 132, 141, 258, 259,
                       318, 395, 463, 453, 475, 455, 489, 415, 362]
        #end
    #end
    
    if layer == '0':
        if model == 'greedy':
            indices = [10, 9, 34, 52, 17, 20, 89, 57, 91, 104, 178, 166, 209, 299, 280, 
                       246, 277, 307, 329, 482, 406, 309, 448, 405, 417]
        elif model == 'iterative':
            indices = [7, 0, 29, 41, 94, 99, 41, 57, 24, 133, 150, 187, 157, 110, 177,
                       153, 254, 294, 297, 238, 237, 315, 442, 441, 411]
            #end
        #end
    #end
    
    
    hidden_dim = int(np.sqrt(W.shape[1]))
    xdim = 5
    ydim = 5
    # if layer == '0' or layer == '1':
    #     indices = list(range(xdim * ydim))
    
    fig = plt.figure(figsize = (5,5))
    for idx, j in zip(indices, list(range(xdim * ydim))):
        ax = fig.add_subplot(ydim, xdim, j+1, xticks = [], yticks = [])
        ax.imshow(W[idx,:].view(hidden_dim, hidden_dim),cmap = 'gray')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #end
    
    
    if plot_specs['save_pictures']:
        dropt = '_' + str(plot_specs['dropout']).replace('.','d') if plot_specs['dropout'] < 1 else '_nodrop'
        gcorr = '' if plot_specs['gcorrection'] == 1 else '_' + str(plot_specs['gcorrection']).replace('.', 'd')
        pictitle = r'{}_receptive_fields_layer{}_cd{}{}{}.pdf'.format(model, layer, plot_specs['mcmc_steps'], dropt, gcorr)
        plt.savefig(plot_specs['path_images'] + r'/{}/receptive_fields/{}'.format(plot_specs['scheme'], pictitle), 
                    dpi = 300, format = 'pdf', bbox_inches = 'tight')
    #end
    plt.show()
    plt.close('all')
#end



def allProfiles(schemes, algs, plot_specs, readout_range = [0.9, 0.98], cost_range = [0.0, 0.04], metrics = ['readout', 'cost']):
    from utls import fetch

    kwords = {'greedy' : 'Greedy',
              'iterative' : 'Iterative',
              'normal' : 'Normal',
              'glorot' : 'Glorot',
              'readout' : 'Readout',
              'cost' : 'MSE'}
              
    kwcolors = {'greedy' : 'g',
                'iterative' : 'r',
                'greedy normal' : 'g',
                'greedy glorot' : 'springgreen',
                'iterative normal' : 'r',
                'iterative glorot' : 'orangered'}
                
    kwline = {'normal' : 'solid',
              'glorot' : '--'}
                
    runs = plot_specs['runs']
    epochs = plot_specs['epochs']
    layers = plot_specs['layers']
    pathmodel = plot_specs['path_models']
    
    readout = {}
    cost    = {}
    dropt = '_' + str(plot_specs['dropout']).replace('.','d') if plot_specs['dropout'] < 1 else '_nodrop'
    gcorr = '_nogcorr' if plot_specs['gcorrection'] == 1 else '_gcorr' + str(plot_specs['gcorrection']).replace('.', 'd')
    
    for alg in algs:
        for scheme in schemes:
            
            readout[(alg, scheme)] = [pd.DataFrame(columns = range(epochs), index = range(runs)) for _ in range(layers)]
            cost[(alg, scheme)]    = [pd.DataFrame(columns = range(epochs), index = range(runs)) for _ in range(layers)]
            path_model = pathmodel + r'/series_{}runs_{}'.format(runs, scheme)
            
            for run in range(runs):
                
                dbn = fetch(path_model + r'/{}_{}_{}_dbn{}_cd{}_ep{}{}{}.pkl'.format(
                             plot_specs['dataset_id'], scheme, alg, run, 
                             plot_specs['mcmc_steps'], 
                             plot_specs['epochs'],
                             dropt,
                             gcorr
                           ))
                
                for i in range(layers):
                    
                    if 'readout' in metrics: readout[(alg, scheme)][i].loc[run] = np.array(dbn.rbm_layers[i].readout_profile, dtype = np.float64)
                    cost[(alg, scheme)][i].loc[run]    = np.array(dbn.rbm_layers[i].cost_profile, dtype = np.float64)
                #end
            #end
        #end
    #end
    
    for metric in metrics:
        
        fig, axs = plt.subplots(1,layers, sharey = False, figsize = (8.5, 2.75))
        for i in range(layers):
        
            ax = axs[i]
            
            for scheme in schemes:
                for alg in algs:
                
                    y = readout[(alg, scheme)][i] if metric == 'readout' else cost[(alg, scheme)][i]
                    
                    ax.errorbar(np.arange(1, epochs+1), y.mean(), #yerr = y.std(), elinewidth = 0.85
                                color = kwcolors['{} {}'.format(alg, scheme)], linestyle = kwline['{}'.format(scheme)],
                                alpha = 0.85, lw = 2, label = '{} {}'.format(kwords[alg], kwords[scheme]))
                    #end
                #end
            #end
            
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
            #end
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
            #end
            ax.set_title('{} Layer {}'.format(kwords[metric], i), fontsize = 14)
            ax.set_xlabel('Epochs', fontsize = 14)
            ax.set_xticks([10,20,30,40,50])
            if metric == 'readout':
                if readout_range.__len__() > 0:
                    ax.set_yticks(list(np.arange(readout_range[0], readout_range[1], 0.01)))
                    ax.set_ylim(readout_range)
                #end
                ax.grid(axis = 'y')
            elif metric == 'cost':
                if cost_range.__len__():
                    ax.set_yticks(list(np.arange(cost_range[0], cost_range[1], 0.01)))
                    ax.set_ylim(cost_range[0], cost_range[1])
                #end
                ax.grid(axis = 'y')
            ax.locator_params(nbins = 5)
            ax.tick_params(axis = 'both', labelsize = 14)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            legloc = 'best' if metric == 'readout' else 'best'
            ax.legend(loc = legloc)
            
        #end
        
        fig.tight_layout()
        if plot_specs['save_pictures']:
            gcorr = '' if plot_specs['gcorrection'] == 1 else '_' + str(plot_specs['gcorrection']).replace('.', 'd')
            pictitle = r'joint_p_{}_profiles_cd{}{}{}.pdf'.format(metric, plot_specs['mcmc_steps'], dropt, gcorr)
            fig.savefig(plot_specs['path_images'] + r'/both/profiles_and_errors/{}'.format(pictitle), 
                        dpi = 300, format = 'pdf', bbox_inches = 'tight')
        
        plt.show(fig)
    #end
#end




def plot_reconstruction_errors(mse_greedy, mse_iterative, plot_specs):
    
    df = pd.DataFrame(columns = ['Greedy', 'Iterative'], index = ['Reproduction', 'Recreation', 'Denoising'])
    df['Greedy'] = mse_greedy.mean(0)
    df['Iterative'] = mse_iterative.mean(0)
    df.plot(kind = 'bar', yerr = [mse_greedy.std(0), mse_iterative.std(0)], 
            alpha = 0.65, color = ['g', 'r'], figsize = (3.5,2.5))
    ax = plt.gca()
    ax.set_ylabel('MSE')
    for tk in ax.get_xticklabels():
        tk.set_rotation(0)
    #end
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    #end
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    #end
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_title('MSE errors for different inference tasks', fontsize = 14)
    
    if plot_specs['save_pictures']:
        dropt = '_' + str(plot_specs['dropout']).replace('.','d') if plot_specs['dropout'] < 1 else '_nodrop'
        gcorr = '' if plot_specs['gcorrection'] == 1 else '_' + str(plot_specs['gcorrection']).replace('.', 'd')
        pictitle = r'mses_cd{}_loops{}{}{}.pdf'.format(plot_specs['mcmc_steps'], plot_specs['loops'], dropt, gcorr)
        plt.savefig(plot_specs['path_images'] + r'/{}/profiles_and_errors/{}'.format(plot_specs['scheme'], pictitle), 
                    dpi = 300, format = 'pdf', bbox_inches = 'tight')
    #end
    plt.show()
    plt.close('all')
#end



















def parameters_histograms(w, dw, a, da, b, db):
    """
    As pointed out in Hinton (2010), a good sanity check to monitor
    the training process is to inspect the parameters -and variations-
    histograms. 
    
    Input:
        ~ X, dX (torch.Tensor) : quantities to plot the histograms of
        X = weights, visible bias and hidden bias
    
    Returns:
        ~ nothing
    """
    w = w.cpu()
    dw = dw.cpu()
    a = a.cpu()
    da = da.cpu()
    b = b.cpu()
    db = db.cpu()
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(231)
    ax.hist(w.reshape(1, w.shape[0] * w.shape[1]))
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Weights', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(232)
    ax.hist(dw.reshape(1, dw.shape[0] * dw.shape[1]))
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Weights variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(233)
    ax.hist(a)
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Visible bias', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(234)
    ax.hist(da)
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Visible bias variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(235)
    ax.hist(b)
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Hidden bias', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(236)
    ax.hist(db)
    ax.locator_params(axis = 'x', nbins = 3)
    ax.set_title('Hidden bias variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.subplots_adjust(hspace=0.25)
    plt.show()
    plt.close('all')
#end


def plot_single_image(image):
    """
    A single digit is displayed.
    Almost useless
    
    Input:
        ~ image (torch.Tensor) : data
        
    Returns:
        ~ nothing
    """
    image = image.cpu()
    
    assert type(image) is torch.Tensor, 'Image to plot is not torch.Tensor'
    image_size = int(np.sqrt(image.shape[0]))
    image = image.view(image_size, image_size)
    
    fig = plt.imshow(image, cmap = 'jet')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close('all')
#end


def plot_images_grid_save(images, labels, title, indx, mode, plot_specs):
    """
    Plot a grid of 2x5 digits
    
    Input:
        ~ images (torch.Tensor) : images data
        ~ labels (torch.Tensor) : labels
        ~ title (string) : this function can be used in many scopes, a title may
                           render clearer 
        
    Returns:
        ~ nothing
    """
    images = images.cpu()
    labels = labels.cpu()
    
    assert type(images[0]) is torch.Tensor, 'Image to plot is not torch.Tensor'
    image_size = int(np.sqrt(images[0].shape[0]))
    
    fig = plt.figure(figsize=(5,2.75))
    for idx in range(10):
        ax = fig.add_subplot(2,10/2,idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].view(image_size, image_size), cmap = 'gray')
        label = int(labels[idx].item())
        ax.set_title(label)
    #end
    fig.suptitle(title, fontsize = 14)
    
    if flow['save_pictures']:
        plt.savefig(plot_specs['path_images'] + r'/{}/digits/{}_{}_cd{}_loops{}_indx{}.pdf'.format(plot_specs['scheme'], plot_specs['train_algorithm'], 
                    mode, plot_specs['mcmc_steps'], plot_specs['loops'], indx), dpi = 300, format = 'pdf', bbox_inches = 'tight')
    #end
    plt.show()
    plt.close('all')
#end

    
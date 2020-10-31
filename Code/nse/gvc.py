
'''
Visualization utilities module.
Here the OOP style is used to easily generate the titles of the figures
to save them univocally

Classes: 
    PlotUtils: parent class in which, accoring to the ``flow'' control structure, 
               methods for saving the graphics with the related title are provided
    
    All the other classes have class-specific features and inherit these said methods
'''

import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
import numpy as np


class PlotUtils:
    
    def __init__(self, flow):
        '''
        Initialization.
        Input:
            - flow (dict): program control structure
        Output:
            none
        '''
        
        self.kwords = {'real'      : 'Real',
                       'fake'      : 'Binomial',
                       'greedy'    : 'Greedy',
                       'iterative' : 'Iterative'}
        self.colors = {'greedy'    : 'g',
                       'iterative' : 'r',
                       'fake'      : 'b'}
        self.__path_figures = flow['pathimages']
        self.threshold = flow['threshold']
        self.dropfrac = flow['dropout']
    #end
    
    def getTitle(self, graphs, scale, epoch):
        '''
        To get the title of the graphics
        Input:
            - graphs (list): a list of gsd.Graph objects (child classes)
            - scale (string): scales of the x and y axes, might it be linear or log 
            - epoch (int): the epoch time stamp
        Output:
            - title (string): the title of the graphics, according to whether the graphs are
                              real or random replicas, the scales and the epoch
        '''
        
        chars = set([g.getChar() for g in graphs])
        try:
            assert len(chars) == 1, 'Weight histogram: Graphs characters not homogeneous!'
        except:
            raise RuntimeError
        #end
        
        title_char  = '{}_'.format(list(chars)[0])
        title_alg   = '{}_'.format(graphs[0].getAlg()) if len(set([g.getAlg() for g in graphs])) == 1 else 'both_'
        title_scale = scale['x'] + scale['y'] + '_'
        title_thre  = str(self.threshold).replace('.','d')
        title_epoch = '_{}'.format(epoch) if epoch != '' else ''
        title_drop  = '_{}'.format(self.dropfrac)
        
        return title_char + title_alg + title_scale + title_thre + title_epoch + title_drop
    #end
    
    def savefig(self, fig, title):
        '''
        Save the figure
        Input:
            - fig (matplotlib.pyplot.figure): figure object to save
            - title (string): title of the figure
        Output: 
            none
        '''
        fig.savefig('{}.pdf'.format(title), format = 'pdf', dpi = 300, bbox_inches = 'tight')
    #end
    def getpath(self):
        '''
        Get the path where the figure should be saved
        Input:
            none
        Output:
            - path (string): path where to save the figure
        '''
        return self.__path_figures
    #end
    
#end

class Histogram(PlotUtils):
    def __init__(self, flow):
        super().__init__(flow)
    #end
    
    def plot(self, graphs, nbins, scale, epoch = '', save = True):
        '''
        Plot the histogram of the network weights
        Input:
            - graphs (list): list of gsd.Graph (child classes), the models that we want to plot
                             the weights histograms of
            - nbist (int): number of bins of the histogram
            - epoch (int): number of the epoch time stamp
            - save (bool): whether to save or not the figure
        Output:
            none
        '''
        
        chars = set([g.getChar() for g in graphs])
        try:
            assert len(chars) == 1, 'Weight histogram: Graphs characters not homogeneous!'
        except:
            raise RuntimeError
        #end
        
        fig, ax = plt.subplots(figsize = (5,3))
        for g in graphs:
            w = g.getEdges('pruned' if g.isPruned() else 'full')['weight'].values
            ax.hist(w, bins = nbins, color = self.colors[g.getAlg()], alpha = 0.45, label = self.kwords[g.getAlg()])
            ax.set_xlabel('Weights', fontsize = 14)
        #end
        title = '{} graph{}'.format(self.kwords[g.getChar()], '' if len(graphs) == 1 else 's')
        ax.set_title(title, fontsize = 18)
        ax.set_yscale(scale['y'])
        ax.legend(loc = 'best', prop = {'size' : 8})
        plt.show()
        
        title_kind  = 'whistpruned_' if all([g.isPruned() for g in graphs]) else 'whist_'
        
        title_pic = title_kind + self.getTitle(graphs, scale, epoch)
        if save:
            self.savefig(fig, self.getpath() + '\{}'.format(title_pic))
        #end
        
    #end
#end

class DegreesDistribution(PlotUtils):
    def __init__(self, flow):
        super().__init__(flow)
    #end
    
    def plot(self, graphs, scale, titlename = 'alg', epoch = '', save = True):
        '''
        Plot the degrees distribution of the given graph(s).
        Input:
            - graphs (list): list of gsd.Graph (again, this list comprehends a real and fake graph)
            - scale (dict): contains the x and y scales specifiers, among linear and log
            - titlename (string): if it is set to 'alg', then the title to write on the plot itself 
                                  specifies the algorithm, otherwise (if 'char'), then the title 
                                  reports the nature of the graph instead (real of fake)
            - epoch (int): as above
            - save (bool): as above
        Output: 
            none
        '''
    
        from scipy.stats import poisson
        
        chars = set([g.getChar() for g in graphs])
        try:
            assert len(chars) == 1, 'Weight histogram: Graphs characters not homogeneous!'
        except:
            raise RuntimeError
        #end
        
        fig, axs = plt.subplots(1,len(graphs), figsize = (int(3 * len(graphs)), 2), sharey = True)
        if type(axs) is np.ndarray:
            pass
        else:
            _axlst = []
            _axlst.append(axs)
            axs = _axlst
        #end
        
        _rmse_return = 0.0
        
        for ax,g,i in zip(axs, graphs, range(len(graphs))):
            
            degs = g.getDegrees()['deg'].values
            frac = g.getDegrees()['fraction'].values
            
            if degs.size <= 2:
                print('No degree other than 0 and 1')
                return
            #end
            
            for i in range(degs.size):
                ax.scatter(degs[i], frac[i], marker = 'o', color = self.colors[g.getAlg()], alpha = 0.65, s = 20)
            #end
            # ax.plot(degs, frac, marker = 'o', color = self.colors[g.getAlg()], alpha = 0.65, markersize = 5)
            
            if graphs[0].getChar() == 'real':
                
                m, q = g.powerLawFit(degs, frac)
                ax.plot(degs[1:], g.powerlaw(degs[1:], m, q), color = self.colors[g.getAlg()], lw = 1.5, alpha = 0.5, label = 'Power-law Fit')
                print('\nReal {} network\ngamma = {:.4f}, a = {:.4f}'.format(self.kwords[g.getAlg()], m, q))
            #end
            
            if g.getChar() == 'fake':
                
                k = g.getAverageDegree()
                print('\nFake {} network\n<k> = {:.6f}'.format(self.kwords[g.getAlg()], k))
                ax.plot(degs[1:], poisson.pmf(degs, k)[1:], color = self.colors[g.getAlg()], 
                        marker = '^', markersize = 2.5, lw = 1.5, alpha = 0.35, label = 'Poisson fit')
                rmse = np.sqrt(np.sum(np.power(frac - poisson.pmf(degs, k), 2)))
                _rmse_return = rmse
            
            # if g.getAlg() == 'fake':
                
            #     k = g.getAverageDegree()
            #     ax.plot(degs[1:], poisson.pmf(degs[1:], k), color = 'k', marker = '^', markersize = 2.5, lw = 1.5, alpha = 0.35, label = 'Poisson Fit')
                
            # #end
            ax.set_xscale(scale['x']); ax.set_yscale(scale['y'])
            
            if titlename == 'alg':
                name = self.kwords[g.getAlg()]
            elif titlename == 'char':
                name = self.kwords[g.getChar()]
            #end
            if epoch != '':
                eptit = ', epoch {}'.format(epoch)
            else:
                eptit = ''
            #end
            
            for tk in ax.yaxis.get_major_ticks(): tk.label.set_fontsize(14)
            for tk in ax.xaxis.get_major_ticks(): tk.label.set_fontsize(14)
            ax.set_title('{} graph{}'.format(name, eptit), fontsize = 18)
            ax.set_xlabel('Degree', fontsize = 14)
            if i == 0: ax.set_ylabel('$p_k$', fontsize = 14)
            if scale['x'] == 'linear':
                ax.set_ylim(bottom = frac.min() - 0.1 * frac.min())
            elif scale['x'] == 'log':
                ax.set_xlim(left = 0.75)
            #end
            if scale['y'] == 'log':
                ax.set_ylim(bottom = frac.min() - 0.1 * frac.min(),
                            top    = degs.max() + 0.5)
            #end
            ax.legend(loc = 'best', prop = {'size' : 12})
            
        #end
        plt.subplots_adjust(hspace = 0.01)
        title_kind  = 'ddisst_'
        
        title_pic = title_kind + self.getTitle(graphs, scale, epoch)
        if save:
            self.savefig(fig, self.getpath() + '\{}'.format(title_pic))
        #end
        
        plt.show()
        return _rmse_return
    #end
#end


class SpectralAnalysis(PlotUtils):
    def __init__(self, flow):
        super().__init__(flow)
    #end
    
    def plot(self, graphs, scale, norm = False, epoch = '', save = True):
        '''
        Plot the spectral properties of the adjacency matrix
        Input:
            - graphs (list): as above
            - scale (dict): as above
            - norm (bool): whether or not to normalize the graph laplacian
            - epoch (int): as above
            - save (bool): as above
        Output:
            none
        '''
        
        for graph in graphs:
            w = graph.getEigs(norm)
            
            fig, ax = plt.subplots(figsize = (5,3))
            
            ax.plot(np.arange(1, w.size+1), w, lw = 1.5, color = self.colors[graph.getAlg()],
                    alpha = 0.65, marker = '+', markersize = 5)
            
            ax.set_title('{} graph, epoch {}'.format(self.kwords[graph.getChar()], epoch), fontsize = 18)
            ax.set_xscale(scale['x']); ax.set_yscale(scale['y'])
            ax.set_xlim(left = 0.75)
            ax.set_ylabel('Eigenvalue magnitude', fontsize = 14)
            ax.set_xlabel('Eigenvalue number', fontsize = 14)
            plt.show()
            
            title_kind  = 'sgae_'
            
            title_pic = title_kind + self.getTitle(graphs, scale, epoch)
            if save:
                self.savefig(fig, self.getpath() + '\{}'.format(title_pic))
            #end
        #end
    #end
#end


class DistancesPlot(PlotUtils):
    def __init__(self, flow):
        super().__init__(flow)
    #end
    
    def plot(self, graphs, scale, epoch = '', save = True):
        '''
        Plot the distribution of the geodesic distances.
        Input:
            - graphs (list): as above
            - scale (dict): as above
            - epoch (int): as above
            - save (bool): as above
        Output:
            none
        '''
        
        for graph in graphs:
            d = graph.geoDistDistribution()
            print('Average distance: {:.4f}'.format(d.mean()))
            
            fig, ax = plt.subplots(figsize = (5,3))
            
            bins = np.arange(0, d.max())
            ax.hist(d, bins = bins, density = True, edgecolor = 'w', 
                    linewidth = 1.5, rwidth = 0.5,
                    alpha = 0.65, color = self.colors[graph.getAlg()])
            
            ax.set_title('{} graph, epoch {}'.format(self.kwords[graph.getChar()],
                                               epoch), fontsize = 18)
            ax.set_xscale(scale['x']); ax.set_yscale(scale['y'])
            ax.set_xlabel('Geodesic Distance', fontsize = 14)
            plt.show()
            
            title_kind = 'dists_'
            title_pic = title_kind + self.getTitle(graphs, scale, epoch)
            if save:
                self.savefig(fig, self.getpath() + '\{}'.format(title_pic))
            #end
        #end
    #end
    
#end
















'''
This module contains the class hierarchy used to model the graphs used in this work, 
namely a real graph coming from the DBN model and the random counterpart used to compare
the properties with.
The distinction is crucial, since, accordingly, the graph pruning strategies are different.
The same, however, does not hold for the degree distribution. The same way could be used for
both, modulo small changes, that can be rendered with an if statement.
'''

import os
import sys
sys.path.append(os.getcwd() + r'\..\dbp')

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



class ModelStream:
    '''
    The purpose of this class is to fetch the model.
    It only contains a static utility method
    '''
    
    @staticmethod
    def from_torch(path_model):
        '''
        Fetch the model, previously identified by the flow control structure.
        Input:
            - path_model (string): where the model is saved, comprehending the name
        Output:
            - model_params (list): list of torch.Tensor objects, that is the weights matrices and 
                                   biases of the model, which are used to create the graph itself
        '''
        
        with open(path_model, 'rb') as f:
            dbn = torch.load(f, map_location = 'cpu')
        #end
        f.close()
        
        model_params = []
        layers = dbn.nlayers

        for i in range(layers):
            model_params.append(dbn.rbm_layers[i].W.numpy().transpose())
            model_params.append(dbn.rbm_layers[i].b.numpy().flatten())
        #end
        
        return model_params
    #end
    
#end

class Graph:
    def __init__(self, params, alg, scheme):
        '''
        Initialization of the graph.
        The fields are coded as private attributes which can be fetched with accessor methods.
        Input:
            - params (list): list of torch.Tensor objects (weights and biases). 
            - alg (string): learning algorithm
            - scheme (string): initialization scheme
        '''
        
        self.__alg    = alg
        self.__scheme = scheme
        self.__Layers = {}
        self.__Nodes  = []
        self.__edges_per_layer = []
        
        # print('Graph instanced: {} trained'.format(self.__alg))
        self.createEdges(params)
        self.createNodes()
        self.__is_pruned = False
    #end
    
    def introduce(self):
        '''
        Could be useful to print the identity of the graph, eg whether it is iterative, real, ...
        Input and Output: none
        '''
        print('\n{} {} network'.format(self.getChar(), self.getAlg()))
    #end
    
    def isPruned(self):
        '''
        Tells whether we have pruned the graph yet
        Input and Output: none
        '''
        return self.__is_pruned
    #end
    
    def setToPruned(self):
        '''
        Once we prune the graph, we record this.
        Input and Output: none
        '''
        self.__is_pruned = True
    #end
    
    '''
    Accessor methods.
    Their purpose is rather clear, comments only on salient things
    '''
    def getAlg(self):
        return self.__alg
    #end
    
    def getLayers(self):
        return self.__Layers
    #end
    
    def getNodes(self):
        return self.__Nodes
    #end
    
    def getNumNodes(self):
        return len(self.__Nodes)
    #end
    
    def getEdges(self, status):
        '''
        We need to request the edges at some point. We must specify whether we want
        the pruned or unpruned graph edges.
        Input:
            - status (string): full or pruned graph
        Output:
            - edges_full or edges_pruned (pandas.DataFrame)
        '''
        if status == 'full':
            return self.__edges_full
        elif status == 'pruned':
            return self.__edges_pruned
        #end
    #end
    
    def getNumEdges(self, status):
        if status == 'full':
            return self.__edges_full.shape[0]
        elif status == 'pruned':
            return self.__edges_pruned.shape[0]
        #end
    #end
    
    def setEdgesPruned(self, edges_pruned):
        self.__edges_pruned = edges_pruned
    #end
    
    def getDegrees(self):
        '''
        Degrees are also saved as a pandas.DataFrame,
        useful to eventually get the degrees average
        '''
        return self.__degrees
    #end
    
    def getNodesDegs(self):
        return self.__nodesdegs
    #end
    
    def getDegMat(self):
        return self.__degMat
    #end
    
    def getAdjMat(self):
        return self.__adjMat
    #end
    
    def getAverageDegree(self):
        '''
        See below for details on the computation of the degrees distribution
        '''
        return (self.__degrees['deg'].values * self.__degrees['fraction']).sum()
    #end
    
    def createNodes(self):
        
        layers = self.__Layers
        Nodes  = []
        for layer in layers.values():
            [Nodes.append(node) for node in list(layer)]
        #end
        self.__Nodes     = Nodes
        self.__num_nodes = len(Nodes)
        # print('Done with nodes')
    #end
    
    def createEdges(self, params):
        '''
        Edges are an important feature of a graph. Other than the connectivity information
        they could carry informations about the strength of the edge itself.
        A good idea is to represent the set of edges of the graph as a pandas.DataFrame,
        so that we can save both the connectivity and strength informations and access them easily.
        Input:
            - params (list): list of torch.Tensor objects, see above
        '''
        
        num_layers = int(len(params) / 2) # bias excluded
        num_nodes  = 0
        layers = {}
        offset = 0
        
        for i in range(num_layers):
            j = 2*i
            
            if i == num_layers - 1:
                num_nodes   += (params[j].shape[0] + params[j].shape[1])
                layers[i]   = np.arange(offset, offset + params[j].shape[0])
                layers[i+1] = np.arange(layers[i][-1] + 1, layers[i][-1] + 1 + params[j].shape[1])
            else:
                num_nodes += params[j].shape[0]
                layers[i] = np.arange(offset, offset + params[j].shape[0])
                offset    = layers[i][-1] + 1   
        #end
        
        links = []
        for k in range(len(layers) - 1):
            
            for i in range(layers[k].size):
                weights = np.array([params[2*k][i,j] for j in range(layers[k+1].size)])
                _links  = np.hstack( (np.ones(layers[k+1].size).reshape(-1,1) * layers[k][i], 
                                      layers[k+1].reshape(-1,1),
                                      weights.reshape(-1,1)) )
                links.append(_links)
            #end
        #end
        
        links = np.vstack(links)
        edges = pd.DataFrame(links, columns = ['src', 'trg', 'weight'],
                             index = np.arange(0, links.shape[0]))
        edges = edges.astype({'src' : int, 'trg' : int, 'weight' : float})
        
        self.__Layers = layers
        self.__edges_full = edges
        
        # print('Done with edges')
        
    #end
    
    def computeAdjMatrix(self, edges_status):
        '''
        Computation of the adjacency matrix. The pruning status information is required.
        Note that the adjacency matrix of the unpruned graph is a blocks matrix.
        Input: 
            - edges_status (string): whether we want to compute the adj mat of the 
                                     pruned or unpruned graph
        Output:
            - A (numpy.array): adjacency matrix
        '''
        
        N = self.getNumNodes()
        A = np.zeros((N,N))
        edf = self.getEdges(edges_status)
        
        adj_lists = [edf[edf['src'] == i]['trg'].tolist() for i in range(N)]
        
        for i in range(N):
            idx = adj_lists[i]
            A[i, idx] = 1
            A[idx, i] = 1
        #end
        
        # self.introduce()
        # print('Done with adjacency matrix'.format(self.getAlg(), self.getChar()))
        return A
    #end
    
    def plotSpy(self, mat):
        '''
        If one wants, could observe the sparsity pattern of the adj mat. 
        Again, the unpruned graph yields a full blocks matrix.
        Input:
            - mat (numpy.array): adj mat
        Output:
            none
        '''
        fig, ax = plt.subplots(figsize = (5,5))
        ax.spy(mat, marker = 'o', markersize = 1)
        plt.show()
        del ax, fig
    #end
    
    
    def degreesDistribution(self, p):
        '''
        Evaluation of the degree distribution.
        Note that this method is called once the edges, nodes and layers informations
        have already been gathered and created. Hence the only additional information is 
        the functional form of the distribution to use. IN Zambra, Testolin and Zorzi (in preparation),
        a discussion about the disfferences in terms of the results of the different choices of the 
        distribution is provided.
        Input:
            p (string): the name of the distribution to use
        Output: 
            none
        '''
        
        '''
        Here the pruned graph is requested! The analysis on the unpruned graph 
        would yield the trivial result of a distribution with only three degrees, 
        due to the architecture of the network
        '''
        try:
            edges = self.getEdges('pruned')
        except:
            raise RuntimeError('Graph not pruned yet')
        #end
        
        def getlen(lst):
            return len(lst)
        #end
        
        N = self.getNumNodes()
        A = self.computeAdjMatrix('full')
        
        '''
        Being the graph directed, we account for the in- and out- degrees, and then merge the 
        degree information in a new dataframe.
        The directional information is not that relevant in this analysis.
        trg == target node identifier
        src == source node identifier
        '''
        nodes_outdeg = edges['src'].value_counts()
        nodes_outdeg = pd.DataFrame( np.hstack((np.array(nodes_outdeg.index).reshape(-1,1), 
                                     np.array(nodes_outdeg.values).reshape(-1,1))),
                                     columns = ['node', 'deg'] )
        nodes_indeg = edges['trg'].value_counts()
        nodes_indeg = pd.DataFrame( np.hstack((np.array(nodes_indeg.index).reshape(-1,1), 
                                    np.array(nodes_indeg.values).reshape(-1,1))),
                                    columns = ['node', 'deg'] )
        
        nodes_degrees = pd.concat([nodes_indeg, nodes_outdeg])
        nodes_degrees = nodes_degrees.groupby('node').sum()
        nodes_degrees = nodes_degrees.reset_index()
        
        '''
        We here add the isolated nodes, which are not visible in the edges dataframe, and set them 
        degree to zero. This may seem a secondary detail but it is crucial, since it reveals how many
        isolated nodes one ends up with, and also defines the shape of the distribution
        '''
        for n in self.getNodes():
            if n not in nodes_degrees['node'].tolist():
                nodes_degrees = nodes_degrees.append(pd.DataFrame([[n, 0]], columns = ['node', 'deg']))
            #end
        #end
        
        nodes_degrees = nodes_degrees.reset_index().drop(columns = 'index')
        
        '''
        PMD: Potential Maximum Degree.
        Since the network is constrained to the layer-layer architecture of the DBN (eg, a 
        node of layer 1 CAN NOT be connected to nodes of layer 3, but only to nodes of layer 2).
        Hence the nodes degrees are penalized according to the maximum degree they could have.
        By thus doing, the inequalities between degrees could be perhaps normalized.
        PMD of a node is computed as the number of connections that he would have in an unpruned
        graph configuration. The nodes degrees are then weighted by this PMD value.
        
        Whether the distribution is chosen to account for the total number of nodes or not, another 
        multiplicative factor should be added.
        
        p:      p_k = N_k/N * (1/w_1 * \delta_{1,k} + ... + 1/w_N * \delta_{N,k}) 
                with w_i = 1/A_{i,1} + ... + 1/A_{i,N} and N_k the number of nodes having deg k
                and \delta_{i,k} the Dirac delta, which kills anything but the nodes with degree k
                in the summation. Here we just want to sum the inverses of the weights related to_frame
                nodes having degree k
        
        q:      p_k = (1/w_1 * \delta_{1,k} + ... + 1/w_N * \delta_{N,k})
                Same as above, but without the multiplicative factor N_k/N.
                The main text (Zambra, Testolin and Zorzi) explains deeply this difference
        '''
        nodes_degrees['PMD'] = np.zeros(nodes_degrees.shape[0])
        nodes_degrees['weight'] = np.zeros(nodes_degrees.shape[0])
        
        if type(np.array(nodes_degrees.index)[0]) is np.float64:
            nodes_degrees = np.array(nodes_degrees.index, dtype = np.int64)
        #end
        
        for i in nodes_degrees.index.tolist():
            pmd = np.count_nonzero(A[i,:])
            nodes_degrees.at[i, 'PMD'] = pmd
            nodes_degrees.at[i, 'weight'] = 1 / pmd
        #end
        
        degrees_frac = nodes_degrees.groupby('deg')['weight'].apply(list).to_frame()
        degrees_frac['node'] = nodes_degrees.groupby('deg')['node'].apply(getlen)
        degrees_frac = degrees_frac.reset_index().rename(columns = {'node' : 'numnodes'})
        degrees_frac['fraction'] = np.zeros(degrees_frac.shape[0])
        degrees_frac = degrees_frac.astype({'numnodes' : np.int64})
        
        for i in degrees_frac.index.tolist():
             dotprod = np.dot( np.ones(degrees_frac.at[i, 'numnodes']), degrees_frac.at[i, 'weight'] )
             if p == 'p': nk = degrees_frac.at[i, 'numnodes'] / N
             if p == 'q': nk = 1
             degrees_frac.at[i, 'fraction'] = dotprod * nk
        #end
        
        '''
        The normalization of the pdf can easily be computed numerically
        '''
        degrees_frac['fraction'] = degrees_frac['fraction'] / degrees_frac['fraction'].sum()
        
        for df in [nodes_degrees, degrees_frac]:
            if type(np.array(df.index)) is np.float64:
                df.index = np.array(df.index, dtype = np.int64)
            #end
        #end
        
        # print('Done with degrees distribution')
        
        self.__degrees = degrees_frac
        self.__nodesdegs = nodes_degrees
        
        D = np.zeros((N,N))
        for i in nodes_degrees['node'].values:
            try:
                D[i, i] = int(nodes_degrees.loc[nodes_degrees['node'] == i]['deg'])
            except:
                print('Max deg is 0, degrees matrix is zeros(N,N)')
            #end
        #end
        
        self.__degMat = D
    #end
    
    def getEigs(self, norm = False, get_ccs = False):
        '''
        Compute the eigenvalues and return them, or return the 
        number of connected components.
        Recall that the number of connected components is the number
        of nonzeros in the spectrum of the eigenvalues
        Input:
            - norm (bool): whether to normalize the graph laplacian
            - get_ccs (bool): if set to true, then the method returns
                              the number of connected components, otherwise,
                              the eigenvalues spectrum is returned.
        Output:
            - w (numpy.array): spectrum of the eigenvalues
            - int(w.size - np.count_nonzero(w)) (int): number of connected components
        '''
        
        from scipy.linalg import eig
        
        A = self.computeAdjMatrix('pruned')
        D = self.getDegMat()
        
        if not norm:
            
            L = D - A
        else:
            
            for diag in np.array(np.diag(D), dtype = np.int64):
                if diag != 0:
                    D[diag, diag] = diag**(-0.5)
                #end
            #end
            
            L = D - A
        #end
        
        w, v = eig(L)
        w = np.real(w)
        w[::-1].sort()
        w[w < 1e-10] = 0
        
        # self.introduce()
        # print('Connected components: {}'.format(w.size - np.count_nonzero(w)))
        # print('Multiplicity of eigenvalue 0: {}'.format(np.count_nonzero(w)))
        
        if not get_ccs:
            return w
        else:
            return int(w.size - np.count_nonzero(w))
        #end
    #end
    
    def geoDistDistribution(self, which = 'array', printd = False):
        '''
        Computation of the geodesic distances.
        Input:
            - which (string): if 'array', then the distances are returned as an array, 
                              otherwise, if 'mean', the average distance is returned
            - printd (bool): if set to true, then the average distance is printed
        Output:
            - distances (numpy.array): array of the numerical distances between nodes
            - dmean (numpy.float64): mean distance
        '''
        
        import scipy.sparse.csgraph as csg
        
        A = self.computeAdjMatrix('pruned')
        
        Asparse = csg.csgraph_from_dense(A)
        Dmatrix, predecessors = csg.shortest_path(Asparse, return_predecessors = True, method = 'D')
        Dmatrix = csg.construct_dist_matrix(A, predecessors, directed = False)
        distances = Dmatrix[Dmatrix != np.inf]
        
        # self.introduce()
        # print('Connected components = {}'.format(csg.connected_components(Asparse)))
        # print('Done with distances')
        
        if which == 'array':
            
            return distances
        
        elif which == 'mean':
            
            dmean = distances.mean()
            # if printd: print('<d> = {:.4f}'.format(dmean))
            
            return dmean
        #end
    #end
    
    @staticmethod
    def getFakeGraph_like(graph, alg, scheme, pscale = 1):
        '''
        Given a graph, this method returns a binomial replica, having the same architecture 
        but random connectivity between nodes. 
        Note: we do not specify only the architecture and the probability of existence p and
        we instead feed to this method a real graph entity because the probability of existence
        is indeed given by the number of connections of the real graph provided.
        If we want the random graph to be similar to our random replica, then we want to set 
        the same probability of existence of an edge, which can be empirically computed (for the real
        graph) as the ratio between the actual number of edges (m) and the maximum possible number of edges (M)
        that one could observe if p = 1, that is,
                
                p = m / M
        
        IMPORTANT: this replica retains the architecture of the DBN, that means that 
        nodes are connected only to neighbouring layers!!!
        Input:
            - graph (gsd.RealGraph): graph model to replicate binomially
            - alg (string): learning algorithm
            - scheme (string): initialization scheme
            - pscale (float): a down-scaling factor for the probability of
                              edge existence
        Output:
            fakegraph (gsd.FakeGraph): binomial graph
        
        Here, unlike the other cases before, it is important to specify that the input and output
        arguments are a REAL and FAKE (binomial) graph respectively. gsd.RealGraph and gsd.FakeGraph are
        children classes of the gsd.Graph class.
        '''
        
        # print('\nCreating Fake Binomial graph')
        p = graph.getNumEdges('pruned') / graph.getNumEdges('full')
        # print('Np = {:.3f}'.format(p * graph.getNumNodes()))
        # print('p = {:.6f}'.format(p))
        
        layers = graph.getLayers()
        fake_params = []
        for i in range(len(layers) - 1):
            W = np.random.binomial(1, p = pscale * p * np.ones((layers[i].size, layers[i+1].size)))
            fake_params.append(W)
            fake_params.append(np.zeros((1,layers[i].size)))
        #end
        
        fakegraph = FakeGraph(fake_params, alg, scheme)
        return fakegraph
    #end
    
    @staticmethod
    def getFakeGraph(size, initialization):
        '''
        Unlike the getFakeGraph_like method, in which the binomial replica is produced 
        based on a real graph structure, here we set an arbitrary value for the probability 
        of existence of the edge. This leads to the creation of a fully ``fantasy'' graph. 
        The only realistic thing is the architecture.
        The value of p is arbitrary, meaning that a sensible choice could be done, typically
        values from 0.001 to 0.09 yield sound results. Sound with respected to the real and real-reliant random replicas.
        Input:
            - size (tuple): number of neurons for layer
            - initialization (dict): instructions for initialization of params
        Output:
            - fakegraph (gsd.FakeGraph): random unrealistic graph created
        '''
        
        def init_uniform(pscale, size):
            return np.random.binomial(1, p = pscale * np.random.uniform(0,1, size))
        def init_normal(loc, scale, size):
            return np.random.normal(loc, scale, size)
        def init_glorot(size, loc):
            return np.random.normal(loc, np.sqrt(2 / size[0] + size[1]), size)
        #end
        
        weight_kernel = {'uniform' : init_uniform,
                         'normal'  : init_normal,
                         'glorot'  : init_glorot}
        
        W = []
        for s in size:
            # W.append(np.random.binomial(1, p = pscale * np.random.uniform(0,1, s)))
            W.append(weight_kernel[initialization['name']](size = s, **initialization['args']))
            W.append(np.zeros(s[0]))
        #end
        
        fakegraph = FakeGraph(W, 'fake', '')
        return fakegraph
    #end
#end


class RealGraph(Graph):
    def __init__(self, params, alg, scheme):
        print('\nReal network')
        self.__char = 'real'
        super().__init__(params, alg, scheme)
    #end
    
    def getChar(self):
        return self.__char
    #end
    
    def prune(self, threshold):
        '''
        The real graph, coming from the DBN, has real-valued strength associated
        with the connections, hence it is necessary to provide a threshold value.
        According to this cut-off value, we delete the smaller (in absolute value)
        connections.
        Input:
            - threshold (float): cut-off parameter
        Output:
            none
        '''
        
        edges = self.getEdges('full')
        mask  = (edges['weight'] <= -threshold) | (edges['weight'] >= threshold)
        edges = edges[mask]
        edges = edges.dropna(axis = 0)
        
        self.setToPruned()
        self.setEdgesPruned(edges)
    #end
    
    '''
    Power-law fit functions, to be used for the degrees distribution
    '''
    def powerlaw(self, x, m, q):
        return (q) * (x ** m)
    #end

    def powerLawFit(self, degs, frac):

        x = np.log10(degs[1:])
        y = np.log10(frac[1:])
        
        (m, q) = np.polyfit(x, y, deg = 1)
        return m, 10**q
    #end
            
#end


class FakeGraph(Graph):
    def __init__(self, params, alg, scheme):
        print('\nFake network')
        self.__char = 'fake'
        super().__init__(params, alg, scheme)
    #end
    
    def getChar(self):
        return self.__char
    #end
    
    def prune(self, threshold = None, binary = True):
        '''
        In this case, being the fake graph binomial, the connection strengths
        are binary-valued, 1 is the connection exists, 0 otehrwise.
        Hence it suffices to eliminate the 0-valued edges.
        One ends up with the collection of existing edges.
        Input:
            - threshold (bool): if the fake graph is given real-valued connections,
                                (would not be binomial in this case), then the threshold
                                is required also here
            - binary (bool): If yes, mean that the graph is simply binomial, hence the pruning 
                             stage consists in cutting off the zero edges
        Output:
            none
        '''
        
        edges = self.getEdges('full')
        
        if binary:
            edges = edges[edges['weight'] != 0.0]
        else:
            mask  = (edges['weight'] <= -threshold) | (edges['weight'] >= threshold)
            edges = edges[mask]
            edges = edges.dropna(axis = 0)
        #end
        
        self.setToPruned()
        self.setEdgesPruned(edges)
    #end    
#end

'''
For comparison purposes, here the same analysis (on the degree distribution) is 
repeated on a completely random network. 
'''

import os
import sys
sys.path.append(os.getcwd() + r'\..\dbp')
from importlib import reload
from pathlib import Path
import numpy as np


import gsd, gvc
reload(gsd)
reload(gvc)

path_images = str(Path(os.getcwd()).parent) + r'\images\both\networks'

p = 0.01
# N = 2000

initializer = {'name' : 'uniform', 
               'args' : {'pscale' : p}}
scalesplot = ['log']
flow = {'pathimages' : path_images, 
        'threshold'  : 1, 
        'dropout'    : 0}
runs = 1

rmses = np.zeros(runs)
degrees = []
avdeg = np.zeros(runs)

# rg = gsd.Graph.getFakeGraph([(2000, 1000)], initializer)
rg = gsd.Graph.getFakeGraph([(784,500), (500,500), (500,2000)], initializer)

rg.prune(binary = True)
rg.degreesDistribution(p = 'q')

for xyscale in scalesplot:
    ddist = gvc.DegreesDistribution( flow )
    
    rmse = ddist.plot([rg], scale = {'x' : xyscale, 'y' : xyscale}, save = True)
    print('RMSE = {}'.format(rmse))
#end

print('Empirical average   <k> = {}'.format(rg.getAverageDegree()))
# avdeg[i] = 2 * rg.getNumEdges('pruned') / rg.getNumNodes()
print('Theoretical average <k> = {}'.format(2 * rg.getNumEdges('pruned') / rg.getNumNodes()))


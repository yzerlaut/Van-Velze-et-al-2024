import brian2
import numpy as np
import matplotlib.pylab as plt
import itertools, scipy.special
from scipy.ndimage import gaussian_filter1d

#######################################################
# A large dictionary storing all networks parameters
#######################################################

REC_POPS = ['ThalExc', 'RecExc', 'RecInh']
COLORS = ['tab:blue', 'tab:green', 'tab:red']

AFF_POPS = ['BgExc', 'ExcToThal']

Model = {
    # numbers of neurons in population
    'N_ThalExc':200, 
    'N_RecExc':4000, 'N_RecInh':800, 
    'N_ExcToThal':100, 'N_BgExc':200, 
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # simulation parameters
    'dt':0.1, 
    # connectivity parameters (proba.)
    'p_RecExc_RecExc':0.05, 'p_RecExc_RecInh':0.05, 
    'p_RecInh_RecExc':0.05, 'p_RecInh_RecInh':0.05, 
    'p_ThalExc_RecExc':0.05, 'p_ThalExc_RecInh':0.05, 
    'p_ExcToThal_ThalExc':0.075,
    'p_BgExc_ThalExc':0.05, 
    'p_BgExc_RecExc':0.02, 
    'p_BgExc_RecInh':0.02,
    # Background Act.
    'F_BgExc':10.,
}

## Cellular Properties
for pop in REC_POPS:
    # common for all
    for key, value in zip(['Gl','Cm','Trefrac','El','Vreset'],
                          [ 10., 200.,  5.0,   -70.,  -70.]):
        Model['%s_%s'%(pop,key)]= value
    # pop-specific
    if pop in ['RecInh']:
        # slightly more excitable
        Model['%s_Vthre'%pop]= -53.
    else:
        Model['%s_Vthre'%pop]= -50.

## Synaptic Weights
for pre, post in itertools.product(AFF_POPS+REC_POPS, REC_POPS):
    if pre=='ThalExc':
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif (pre=='BgExc') and ('Rec' in post):
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif 'Exc' in pre:
        Model['Q_%s_%s'%(pre, post)] = 2. # nS
    elif 'Inh' in pre:
        Model['Q_%s_%s'%(pre, post)] = 10. # nS

    

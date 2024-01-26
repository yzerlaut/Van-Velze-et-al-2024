import sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt

from neural_network_dynamics import ntwk

from Modulation import *

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

## Recurrently-connected populations
REC_POPS = ['PyrExc', 'PvInh', 'SstInh', 'VipInh', 'ThalExc']
COLORS = ['tab:green','tab:red','tab:orange','tab:purple','tab:blue']
Model = {
    # numbers of neurons in population
    'N_PyrExc':4000, 'N_PvInh':800, 'N_VipInh':200, 'N_SstInh':200, # cortex
    'N_ThalExc':100,  # thalamus
    # 
    # ---- some other common props ----
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # simulation parameters
    'dt':0.1, 'SEED':3, # low by default, see later
}


## Afferent excitatory populations: Background + Locomotion + Sensory-Drive
AFF_POPS = ['BgExc', 'LocExc', 'SDExc']
for pop in AFF_POPS:
    Model['N_%s'%pop] = 100 # common for all aff pops

## Synaptic Weights
for pre, post in itertools.product(AFF_POPS+REC_POPS, REC_POPS):
    if pre in AFF_POPS+['ThalExc']:
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif 'Exc' in pre:
        Model['Q_%s_%s'%(pre, post)] = 2. # nS
    elif 'Inh' in pre:
        Model['Q_%s_%s'%(pre, post)] = 10. # nS


## Cellular Properties
for pop in REC_POPS:
    # common for all
    Model['%s_Gl'%pop]= 10.
    Model['%s_Cm'%pop]= 200.
    Model['%s_Trefrac'%pop]= 5.
    Model['%s_El'%pop]= -70.
    Model['%s_Vreset'%pop]= -70.
    Model['%s_a'%pop]= 0.
    Model['%s_b'%pop]= 0.
    Model['%s_tauw'%pop]= 1e9
    Model['%s_deltaV'%pop]= 0.
    # pop-specific
    if pop in ['PvInh', 'SstInh', 'VipInh']:
        # slightly more excitable
        Model['%s_Vthre'%pop]= -53.
    else:
        Model['%s_Vthre'%pop]= -50.

## Connectivity Parameters
# background to pop
Model['p_BgExc_ThalExc'] = 0.1
Model['p_BgExc_SstInh'] = 0.1
Model['p_BgExc_VipInh'] = 0.02

# thalamic input to cortex
Model['p_ThalExc_PyrExc'] = 0.1
Model['p_ThalExc_PvInh'] = 0.15

# cortical recurrence
Model['p_PyrExc_PyrExc'] = 0.05
Model['p_PyrExc_PvInh'] = 0.05
Model['p_PvInh_PyrExc'] = 0.05
Model['p_PvInh_PvInh'] = 0.05

# disinhibition
Model['p_VipInh_SstInh'] = 0.15
Model['p_SstInh_VipInh'] = 0.05

# Locomotion
Model['p_LocExc_ThalExc'] = 0.05
Model['p_LocExc_VipInh'] = 0.05


## Background Activity
Model['F_BgExc'] = 15.
Model['F_LocExc'] = 15.
Model['F_SDExc'] = 15.
# 'p_PyrExc_PyrExc':0.02, 'p_PyrExc_Inh':0.02, 
# 'p_PvInh_PyrExc':0.02, 'p_PvInh_PvInh':0.02, 
# 'p_VipInh_SstInh':0.02, 
# 'p_ThalExc_PyrExc':0.1, 'p_ThalExc_Inh':0.1, 'p_ThalExc_DsInh':0.1, 
# # simulation parameters
# 'dt':0.1, 'tstop': 1000., 'SEED':3, # low by default, see later


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('data/test.h5')

    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm(data, 
                                      figsize=(10,2),
                                      COLORS = COLORS)
    
    plt.show()
else:
    # build stimulation
    t, SensoryDrive, Locomotion = build_arrays(props, dt=Model['dt'])
    Model['tstop'] = t[-1]+Model['dt']

    # build recurrent populations
    NTWK = ntwk.build.populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_raster=True, with_Vm=1,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    # # # afferent excitation onto thalamic excitation
    for Aff, AffRate in zip(AFF_POPS, 
                            [1.+0.*t, Locomotion, SensoryDrive]):
        for p, pop in enumerate(REC_POPS):
            if ('p_%s_%s' % (Aff, pop) in Model) and\
                    (Model['p_%s_%s' % (Aff,pop)] > 0):
                ntwk.stim.construct_feedforward_input(NTWK, 
                                                      pop, Aff,
                                                      t, Model['F_%s'%Aff]*AffRate,
                                                      verbose=True,
                                                      SEED=3*p+1)


    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.recording.write_as_hdf5(NTWK, filename='data/test.h5')
    print('Results of the simulation are stored as:', '4pop_model_data.h5')
    print('--> Run \"python 4pop_model.py plot\" to plot the results')

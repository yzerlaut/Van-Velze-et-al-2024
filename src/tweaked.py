import brian2
import numpy as np
import matplotlib.pylab as plt
import itertools, scipy.special
from scipy.ndimage import gaussian_filter1d

#######################################################
# A large dictionary storing all networks parameters
#######################################################

REC_POPS = ['PyrExc', 'PvInh', 'VipInh', 'SstInh']
COLORS = ['tab:green', 'tab:red', 'tab:purple', 'tab:orange']

AFF_POPS = ['AffExc']

Model = {
    # numbers of neurons in population
    'N_PyrExc':4000, 'N_PvInh':1000, 'N_SstInh':200,
    'N_AffExc':100, 'N_VipInh':250,
    # synaptic time constants (ms)
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials (mV)
    'Ee':0., 'Ei': -80.,
    # connectivity parameters (proba.)
    'p_PyrExc_PyrExc':0.05, 'p_PyrExc_PvInh':0.05, 
    'p_PvInh_PyrExc':0.05, 'p_PvInh_PvInh':0.05, 
    'p_VipInh_PvInh':0.1, 
    'p_AffExc_PyrExc':0.1, 'p_AffExc_PvInh':0.1, 
    'p_AffExc_SstInh':0.2, # 'p_VipInh_SstInh':0.1,
    'p_AffExc_VipInh':0.1, # 'p_SstInh_VipInh':0.02,
    # simulation parameters (ms)
    'dt':0.1, 'SEED':3, # low by default, see later
    # === afferent population waveform:
    'Faff1':5.,'Faff2':15.,'Faff3':10.,
    'DT':900., 'rise':50.
}

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
    if pop in ['PvInh']:
        # slightly more excitable
        Model['%s_Vthre'%pop]= -53.
    else:
        Model['%s_Vthre'%pop]= -50.

## Synaptic Weights
for pre, post in itertools.product(AFF_POPS+REC_POPS, REC_POPS):
    if pre in AFF_POPS:
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif 'Exc' in pre:
        Model['Q_%s_%s'%(pre, post)] = 2. # nS
    elif 'Inh' in pre:
        Model['Q_%s_%s'%(pre, post)] = 10. # nS

def build_neuron_params(Model,
                           NRN_KEY, N=1):
    """ we construct a dictionary from the """
    params = {'name':NRN_KEY, 'N':N}
    keys = ['Gl', 'Cm','Trefrac', 'El', 'Vthre', 'Vreset']
    for k in keys:
        params[k] = Model[NRN_KEY+'_'+k]
    return params

def get_membrane_equation(neuron_params, synaptic_array,\
                          verbose=False):
    ## -- membrane equation: Vm dynamics
    eqs = """
    dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + I)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    
    ## -- synaptic currents: 1) adding all synaptic currents to the membrane equation via the I variable
    eqs += """
        I = I0 """
    for synapse in synaptic_array:
        if synapse['pconn']>0:
            # loop over each presynaptic element onto this target
            Gsyn = 'G'+synapse['name']
            eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
    eqs += ' : amp'

    ## --  synaptic currents: 2) constructing the temporal dynamics of the synaptic conductances
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        if synapse['pconn']>0:
            Gsyn = 'G'+synapse['name']
            eqs += """
            """+'d'+Gsyn+'/dt = -'+Gsyn+'*(1./(%(Tsyn)f*ms)) : siemens' % synapse
    eqs += """
        I0 : amp """

    if verbose:
        print('------------------------------------------------------------------')
        print('==> Neuron Type', neuron_params['name'])
        print('--------> with parameters:', neuron_params)
        print('--------> Equations:', eqs)
        
    neurons = brian2.NeuronGroup(neuron_params['N'], model=eqs,
                                 method='euler',
                                 refractory=str(neuron_params['Trefrac'])+'*ms',
                                 threshold='V>'+str(neuron_params['Vthre'])+'*mV',
                                 reset='V='+str(neuron_params['Vreset'])+'*mV')
    return neurons


def get_syn_and_conn_matrix(Model,
                            POPULATIONS,
                            AFFERENT_POPULATIONS=[],
                            verbose=False):

    SOURCE_POPULATIONS = POPULATIONS+AFFERENT_POPULATIONS
    
    # creating empty arry of objects (future dictionnaries)
    M = np.empty((len(SOURCE_POPULATIONS), len(POPULATIONS)), dtype=object)
    # default initialisation
    for i, j in itertools.product(range(len(SOURCE_POPULATIONS)), range(len(POPULATIONS))):
        source_pop, target_pop = SOURCE_POPULATIONS[i], POPULATIONS[j]
        if 'Exc' in source_pop:
            Erev, Ts = Model['Ee'], Model['Tse'] # common Erev and Tsyn to all excitatory currents
        elif 'Inh' in source_pop:
            Erev, Ts = Model['Ei'], Model['Tsi'] # common Erev and Tsyn to all inhibitory currents
        else:
            print(' /!\ SOURCE POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev, Ts = Model['Ee'], Model['Tse']

        # CONNECTION PROBABILITY AND SYNAPTIC WEIGHTS
        if ('p_'+source_pop+'_'+target_pop in Model.keys()) and ('Q_'+source_pop+'_'+target_pop in Model.keys()):
            pconn, Qsyn = Model['p_'+source_pop+'_'+target_pop], Model['Q_'+source_pop+'_'+target_pop]
        else:
            if verbose:
                print('No connection for:', source_pop,'->', target_pop)
            pconn, Qsyn = 0., 0.
                
        M[i, j] = {'pconn': pconn, 'Q': Qsyn,
                   'Erev': Erev, 'Tsyn': Ts,
                   'name':source_pop+target_pop}

    return M

def build_populations(Model,
                      POPULATIONS,
                      AFFERENT_POPULATIONS=[],
                      with_Vm=2,
                      verbose=False):
    """
    sets up the neuronal populations
    and construct a network object containing everything: NTWK
    """

    ## NEURONS AND CONNECTIVITY MATRIX
    NEURONS = []
    for pop in POPULATIONS:
        NEURONS.append({'name':pop, 'N':Model['N_'+pop]})

    NTWK = {'NEURONS':NEURONS, 'Model':Model,
            'POPULATIONS':np.array(POPULATIONS),
            'M':get_syn_and_conn_matrix(Model,
                                        POPULATIONS,
                                        AFFERENT_POPULATIONS=AFFERENT_POPULATIONS,
                                        verbose=verbose)}
    
    ########################################################################
    ####  Setting up 
    ########################################################################
    
    NTWK['POPS'] = []
    for ii, nrn in enumerate(NEURONS):
        neuron_params = build_neuron_params(Model,
                                            nrn['name'], N=nrn['N'])
        NTWK['POPS'].append(get_membrane_equation(neuron_params,
                                                  NTWK['M'][:,ii],
                                                  verbose=verbose))
        nrn['params'] = neuron_params

    ########################################################################
    #### Recordings
    ########################################################################
    
    NTWK['POP_ACT'] = []
    for pop in NTWK['POPS']:
        NTWK['POP_ACT'].append(brian2.PopulationRateMonitor(pop))

    NTWK['RASTER'] = []
    for pop in NTWK['POPS']:
        NTWK['RASTER'].append(brian2.SpikeMonitor(pop))
        
    if with_Vm>0:
        NTWK['VMS'] = []
        for pop in NTWK['POPS']:
            NTWK['VMS'].append(brian2.StateMonitor(pop, 'V', record=np.arange(with_Vm)))
            
    NTWK['PRE_SPIKES'], NTWK['PRE_SYNAPSES'] = [], [] # for future afferent inputs
    
    return NTWK

def build_up_recurrent_connections(NTWK, SEED=1, verbose=False):
    """
    Construct the synapses from the connectivity matrix 
    """
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    np.random.seed(SEED)

    if verbose:
        print('------------------------------------------------------')
        print('drawing random connections [...]')
        print('------------------------------------------------------')
        
    for ii, jj in itertools.product(range(len(NTWK['POPS'])), range(len(NTWK['POPS']))):
        if (NTWK['M'][ii,jj]['pconn']>0) and (NTWK['M'][ii,jj]['Q']!=0):
            CONN[ii,jj] = brian2.Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj], model='w:siemens',\
                               on_pre='G'+NTWK['M'][ii,jj]['name']+'_post+=w')
            # N.B. the following brian2 settings:
            # CONN[ii,jj].connect(p=NTWK['M'][ii,jj]['pconn'], condition='i!=j')
            # does not fix synaptic numbers, so we draw manually the connections
            N_per_cell = int(NTWK['M'][ii,jj]['pconn']*NTWK['POPS'][ii].N)
            if ii==jj: # need to take care of no autapse
                i_rdms = np.concatenate([\
                                np.random.choice(
                                    np.delete(np.arange(NTWK['POPS'][ii].N), [iii]), N_per_cell)\
                                          for iii in range(NTWK['POPS'][jj].N)])
            else:
                i_rdms = np.concatenate([\
                                np.random.choice(np.arange(NTWK['POPS'][ii].N), N_per_cell)\
                                          for jjj in range(NTWK['POPS'][jj].N)])
            j_fixed = np.concatenate([np.ones(N_per_cell,dtype=int)*jjj for jjj in range(NTWK['POPS'][jj].N)])
            CONN[ii,jj].connect(i=i_rdms, j=j_fixed) 
            CONN[ii,jj].w = NTWK['M'][ii,jj]['Q']*brian2.nS
            CONN2.append(CONN[ii,jj])

    NTWK['REC_SYNAPSES'] = CONN2


# #######################################################
# ################## AFFERENT INPUTS ####################
# #######################################################

def waveform(t, Model):
    waveform = 0*t
    # first waveform
    for tt, fa in zip(\
         2.*Model['rise']+np.arange(3)*(3.*Model['rise']+Model['DT']),
                      [Model['Faff1'], Model['Faff2'], Model['Faff3']]):
        waveform += fa*\
             (1+scipy.special.erf((t-tt)/Model['rise']))*\
             (1+scipy.special.erf(-(t-tt-Model['DT'])/Model['rise']))/4
    return waveform

def set_spikes_from_time_varying_rate(time_array, rate_array, N, Nsyn, SEED=1):
    """
    generates an inhomogeneous Poisson process from a time-varying waveform in Hz
    """
    np.random.seed(SEED) # setting the seed !
    
    ## time_array in ms !!
    # so multplying rate array
    
    indices, times = [], []
    DT = (time_array[1]-time_array[0])
    
    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N)
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]:
            indices.append(ii) # all the indices
            times.append(time_array[it]) # all the same time !

    return np.array(indices), np.array(times)*brian2.ms

def construct_feedforward_input(NTWK,
                                target_pop, afferent_pop,\
                                t, rate_array,\
                                verbose=False,
                                SEED=1):
    """
    This generates an input asynchronous from post synaptic neurons to post-synaptic neurons

    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!
    """

    Model = NTWK['Model']
    
    # extract parameters of the afferent input
    Nsyn = Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+afferent_pop]
    Qsyn = Model['Q_'+afferent_pop+'_'+target_pop]

    #finding the target pop in the brian2 objects
    ipop = np.argwhere(NTWK['POPULATIONS']==target_pop).flatten()[0]
    
    if Nsyn>0:
        if verbose:
            print('drawing Poisson process for afferent input [...]')
        indices, times = set_spikes_from_time_varying_rate(\
                            t, rate_array,\
                            NTWK['POPS'][ipop].N, Nsyn, SEED=(SEED+2)**2%100)
        spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times)
        pre_increment = 'G'+afferent_pop+target_pop+' += w'
        synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre=pre_increment,\
                                        model='w:siemens')
        synapse.connect('i==j')
        synapse.w = Qsyn*brian2.nS

        NTWK['PRE_SPIKES'].append(spikes)
        NTWK['PRE_SYNAPSES'].append(synapse)
        
    else:
        print('Nsyn = 0 for', afferent_pop+'_'+target_pop)
    
            
# ################################################################
# ## --------------- Initial Condition ------------------------ ##
# ################################################################

def initialize_to_rest(NTWK):
    """
    Vm to resting potential and conductances to 0
    """
    for ii in range(len(NTWK['POPS'])):
        NTWK['POPS'][ii].V = NTWK['NEURONS'][ii]['params']['El']*brian2.mV
        for jj in range(len(NTWK['POPS'])):
            if NTWK['M'][jj,ii]['pconn']>0: # if connection
                exec("NTWK['POPS'][ii].G"+NTWK['M'][jj,ii]['name']+" = 0.*brian2.nS")
                
# #####################
# ## ----- Run ----- ##
# #####################


def collect_and_run(NTWK, verbose=False):
    """
    collecting all the Brian2 objects and running the simulation
    """
    NTWK['dt'], NTWK['tstop'] = NTWK['Model']['dt'], NTWK['Model']['tstop'] 
    brian2.defaultclock.dt = NTWK['dt']*brian2.ms
    net = brian2.Network(brian2.collect())
    OBJECT_LIST = []
    for key in ['POPS',
                'REC_SYNAPSES', 'RASTER',
                'POP_ACT', 'VMS',
                'PRE_SPIKES', 'PRE_SYNAPSES']:
        if key in NTWK.keys():
            net.add(NTWK[key])

    print('running simulation [...]')
    net.run(NTWK['tstop']*brian2.ms)
    return net

def run_3pop_ntwk_model(Model,
                        filename='data/sas.h5',
                        with_Vm=4,
                        verbose=False,
                        SEED=3):

    print('initializing simulation [...]')
    NTWK = build_populations(Model,
                             REC_POPS,
                             AFFERENT_POPULATIONS=['AffExc'],
                             with_Vm=with_Vm,
                             verbose=verbose)

    build_up_recurrent_connections(NTWK,
                                   SEED=Model['SEED'], 
                                   verbose=verbose)


    Model['tstop'] = Model['rise']+3*(3.*Model['rise']+Model['DT'])
    NTWK['t_array'] = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    NTWK['faff_waveform'] = waveform(NTWK['t_array'], Model)

    for Aff in AFF_POPS:
        for i, tpop in enumerate(REC_POPS): 
            if ('p_%s_%s' % (Aff, pop) in Model) and\
                    (Model['p_%s_%s' % (Aff,pop)] > 0):
                construct_feedforward_input(NTWK, tpop, Aff,
                                            NTWK['t_array'],
                                            NTWK['faff_waveform'],
                                            verbose=verbose,
                                            SEED=int(37*SEED+i)%13)

    initialize_to_rest(NTWK)
    
    network_sim = collect_and_run(NTWK,
                                  verbose=verbose)
    
    print('-> done !')

    return NTWK
    

if __name__=='__main__':
    
    import argparse
    parser=argparse.ArgumentParser(description="""
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-v", "--verbose", help="print stuff",
                        action="store_true")
    
    args = parser.parse_args()
    NTWK = run_3pop_ntwk_model(Model,
                               with_Vm=3,
                               verbose=args.verbose)

    ### PLOT ###
    fig = plt.figure(figsize=(7,5.5))
    plt.subplots_adjust()
    # afferent stimulation
    ax1 = plt.subplot2grid((6,1), (0,0))
    ax1.plot(NTWK['t_array'], NTWK['faff_waveform'], 'k-')
    ax1.set_xticks([]);ax1.set_ylabel(r'$\nu_a$ (Hz)')
    # populations activity (instant. firing rates)
    ax2 = plt.subplot2grid((6,1), (1, 0), rowspan=2)
    for i, pop in enumerate(REC_POPS):
        rate = NTWK['POP_ACT'][i].rate/brian2.Hz
        rate = gaussian_filter1d(rate, int(20./0.1)) # smoothing
        rate[rate<0.01] = 0.01
        # ax2.semilogy(NTWK['t_array'], rate, '-', color=COLORS[i], label=pop)
        ax2.plot(NTWK['t_array'], rate, '-', color=COLORS[i], label=pop)
    ax2.legend(frameon=False)
    ax2.set_xticks([]);ax2.set_ylabel('pop act. (Hz)')
    # sample Vm traces 
    ax3 = plt.subplot2grid((6,1), (3, 0), rowspan=3)
    N = [3,1,1,1] # number displayed per population
    j=0 # index to shift the Vm trace
    for i, pop in enumerate(REC_POPS):
        for n in range(N[i]):
            ax3.plot(NTWK['t_array'], NTWK['VMS'][i].V[n]/brian2.mV-20*j,
                     '-', color=COLORS[i])
            j+=1
    ax3.plot([0.09, 0.09], [-60, -50], 'k-')
    ax3.annotate('10mV', (0.1, -70))
    ax3.set_yticks([]);ax3.set_ylabel('sample Vm traces')
    ax3.set_xlabel('time (ms)')
    plt.show()
    

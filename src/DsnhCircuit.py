import itertools

#####################################################
#  A dictionary storing all networks parameters     #
#####################################################

REC_POPS = ['VipInh', 'SstInh']
COLORS = ['tab:purple', 'tab:orange']

AFF_POPS = ['BgExc', 'ExcToVip']

Model = {
    # numbers of neurons in population
    'N_SstInh':100, 'N_VipInh':100,
    'N_ExcToVip':100, 'N_BgExc':200, 
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # simulation parameters
    'dt':0.1, 
    # connectivity parameters (proba.)
    # Vip -> Sst
    'p_VipInh_SstInh':0.3, 
    'p_SstInh_VipInh':0.025, 
    # Exc -> Sst
    'p_BgExc_SstInh':0.15, 
    # Exc -> Vip
    'p_BgExc_VipInh':0.1, 
    'p_ExcToVip_VipInh':0.05, 
    # Background Act.
    'F_BgExc':10.,
}

## Cellular Properties
for pop in REC_POPS:
    # common for all
    for key, value in zip(['Gl','Cm','Trefrac','El','Vreset','Vthre'],
                          [ 10., 200.,  5.0,   -70.,  -70.,   -53.]):
        Model['%s_%s'%(pop,key)]= value

## Synaptic Weights
for pre, post in itertools.product(AFF_POPS+REC_POPS, REC_POPS):
    if pre=='ThalExc':
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif 'Exc' in pre:
        Model['Q_%s_%s'%(pre, post)] = 2. # nS
    elif 'Inh' in pre:
        Model['Q_%s_%s'%(pre, post)] = 10. # nS
    else:
        raise BaseException(' pop. not recognized !')


if __name__=='__main__':
    
    import numpy as np
    import matplotlib.pylab as plt

    from ntwk_sim import *

    import sys
    sys.path.append('./')
    from neural_network_dynamics import ntwk

    import argparse
    parser=argparse.ArgumentParser(description="""
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-v", "--verbose", help="print stuff",
                        action="store_true")
   
    Model['tstop'] = 1e3

    t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    args = parser.parse_args()
    NTWK = run_ntwk_sim(Model, REC_POPS,
                        AFF_POPS=AFF_POPS,
                        AFF_RATE_ARRAYS = [30.+0*t, 0*t],
                        verbose=args.verbose)
    ntwk.recording.write_as_hdf5(NTWK, filename='data/test.h5')

    data = ntwk.recording.load_dict_from_hdf5('data/test.h5')
    fig, _ = ntwk.plots.raster_and_Vm(data, REC_POPS,
                                      figsize=(10,2),
                                      COLORS = COLORS)
    plt.show()

import brian2
import numpy as np
import matplotlib.pylab as plt
import itertools, scipy.special
from scipy.ndimage import gaussian_filter1d

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from ntwk_sim import *
import sys
sys.path.append('./')
from neural_network_dynamics import ntwk

dt = 0.1

Fmin, Fmax, N = 0., 12., 13
episode, inter_frac = 1000., 0.2
smoothing = 50.
Inputs = Fmin+np.arange(N)/(N-1)*(Fmax-Fmin)

t = np.arange(int(episode*N/dt))*dt
stim = 0*t
for e in range(N):
    t0 = e*episode+episode*inter_frac
    t1 = e*episode+episode*(1-inter_frac)
    stim[(t>t0) & (t<t1)] = Inputs[e]
stim = gaussian_filter1d(stim, int(smoothing/dt))

if sys.argv[1]=='plot':

    data = ntwk.recording.load_dict_from_hdf5('data/IO-Dsnh.h5')
    fig, _ = ntwk.plots.raster_and_Vm(data, REC_POPS,
                                      figsize=(10,2),
                                      COLORS = COLORS,
                                      Vm_plot_args=dict(clip_spikes=True))

    plt.show()


elif sys.argv[1]=='Disinhibitory':

    from DsnhCircuit import *

    Model['dt'], Model['tstop'] = dt, t[-1]+dt

    NTWK = run_ntwk_sim(Model, REC_POPS,
                        AFF_POPS=AFF_POPS,
                        AFF_RATE_ARRAYS = [Model['F_BgExc']+0*t, 
                                           stim])
    ntwk.recording.write_as_hdf5(NTWK, filename='data/IO-Dsnh.h5')

elif sys.argv[1]=='Recurrent':

    from RecCircuit import *

    Model['dt'], Model['tstop'] = dt, t[-1]+dt

    NTWK = run_ntwk_sim(Model, REC_POPS,
                        AFF_POPS=AFF_POPS,
                        AFF_RATE_ARRAYS = [Model['F_BgExc']+0*t, 
                                           stim])
    ntwk.recording.write_as_hdf5(NTWK, filename='data/IO-Rec.h5')

elif 'Both' in sys.argv[1]:


    from CoupledCircuit import *

    if 'Coupled' in sys.argv[1]:
        filename = 'data/IO-Both-Coupled.h5'
    else:
        print(Model['p_RecExc_SstInh'])
        Model['p_RecExc_SstInh'] = 0
        print(Model['p_RecExc_SstInh'])
        filename = 'data/IO-Both-Uncoupled.h5'
        
    Model['dt'], Model['tstop'] = dt, t[-1]+dt

    NTWK = run_ntwk_sim(Model, REC_POPS,
                        AFF_POPS=AFF_POPS,
                        AFF_RATE_ARRAYS = [Model['F_BgExc']+0*t, 
                                           stim])

    ntwk.recording.write_as_hdf5(NTWK, filename=filename)


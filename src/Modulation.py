import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pylab as plt

EpisodeLength = 2e3
Fraction = 0.4
Smoothing = 50

whisking_event = {'fraction':Fraction,
                  'amplitude':4.0,
                  'name':'whisking',
                  'color':'dodgerblue'}
running_event = {'fraction':Fraction,
                 'amplitude':6.,
                  'name':'running',
                 'color':'firebrick'}
whiskerAff_event = {'fraction':Fraction,
                    'amplitude':4.0,
                    'name':'whisk. stim.',
                    'color':'darkturquoise'}
light_event = {'fraction':1.0,
               'amplitude':4.0,
               'name':'light stim.',
               'color':'tab:olive'}

Types = [whisking_event, running_event,
         whiskerAff_event, light_event]

Events = [\
    {'name':'sensory drive \n mid-arousal event',
     'neuromodulatory':[whisking_event],
     # 'sensory':[light_event],
     'sensory':[whiskerAff_event],
     },
    {'name':'sensory drive \n high-arousal event ',
     'neuromodulatory':[whisking_event, running_event],
     # 'sensory':[light_event],
     'sensory':[whiskerAff_event],
     },
    {'name':' no sensory drive \n high-arousal event',
     'neuromodulatory':[whisking_event, running_event],
     'sensory':[],
     }
]

def make_increment(t, e, fraction, 
                   length, amplitude,
                   smoothing, dt):
    tcenter = e*length+length/2.
    increment = 0*t
    cond = (t>(tcenter-length*fraction/2.)) &\
        (t<(tcenter+length*fraction/2.))
    increment[cond] += amplitude
    return gaussian_filter1d(increment, int(smoothing/dt))


def build_arrays(Events,
                 length = EpisodeLength,
                 dt=0.1,
                 smoothing=Smoothing,
                 AX = None):


    Nep = len(Events)
    tfull = Nep*length
    t = np.arange(int(tfull/dt))*dt

    Neuromodulatory, Sensory = 0*t, 0*t

    for e, Event in enumerate(Events):
        for i, key, array in zip(range(2),
                                 ['neuromodulatory', 'sensory'],
                                 [Neuromodulatory, Sensory]):
            for event in Event[key]:
                new = make_increment(t, e, event['fraction'], 
                                     length, event['amplitude'],
                                     smoothing, dt) 
                if AX is not None:
                    AX[i].fill_between(t[::50], 
                                       array[::50], 
                                       array[::50]+new[::50], 
                                       lw=0, alpha=0.7, 
                                       color=event['color'])
                array += new

    if AX is not None:
        for e, Event in enumerate(Events):
            t0 = e*length+length/2.
            AX[1].annotate('\n'+Event['name'], (t0, 0), 
                           xycoords='data', va='top', ha='center')

    return t, Neuromodulatory, Sensory

if __name__=='__main__':

    import matplotlib.pylab as plt
    fig, AX = plt.subplots(2, 1, figsize=(10,2))
    plt.subplots_adjust(bottom=.4, top=.99)
    _ = build_arrays(Events, AX=AX)
    plt.show()


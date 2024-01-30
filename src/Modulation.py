import numpy as np
from scipy.ndimage import gaussian_filter1d

loc_amp = 2.5
sd_amp = 3.

props = {
    'episode':600, # in ms
    'V1-dark-locomotion':{'t0_SD':0, # in ms
                          'amp_SD':0, # in Hz
                          'duration_SD':0., # in ms
                          't0_Loc': 200, # in ms
                          'amp_Loc':loc_amp, # in Hz
                          'duration_Loc':200},
    'V1-light-locomotion':{'t0_SD':50, # in ms
                          'amp_SD':sd_amp, # in Hz
                          'duration_SD':500, # in ms
                          't0_Loc': 200, # in ms
                          'amp_Loc':loc_amp, # in Hz
                          'duration_Loc':200},
    'S1-trimmed-locomotion':{'t0_SD':0, # in ms
                             'amp_SD':sd_amp, # in Hz
                             'duration_SD':0, # in ms
                             't0_Loc': 200, # in ms
                             'amp_Loc':loc_amp, # in Hz
                             'duration_Loc':200},
    'S1-whisking-locomotion':{'t0_SD':200, # in ms
                              'amp_SD':sd_amp, # in Hz
                              'duration_SD':200, # in ms
                              't0_Loc': 200, # in ms
                              'amp_Loc':loc_amp, # in Hz
                              'duration_Loc':200},
    'S1-whisking-only':{'t0_SD':200, # in ms
                              'amp_SD':sd_amp, # in Hz
                              'duration_SD':200, # in ms
                              't0_Loc': 0, # in ms
                              'amp_Loc':0, # in Hz
                              'duration_Loc':0},
}
    
    
                          
                          
def build_arrays(props,
                 dt=0.1,
                 pre_time = 200,
                 smoothing=50,
                 time_factor=2):

    Nep = len(props.keys())-1
    tfull = time_factor*(pre_time+Nep*props['episode'])
    t = np.arange(int(tfull/dt))*dt
    SD, Loc = 0*t, 0*t
    for i, episode in enumerate(list(props.keys())[1:]):
        for key, array in zip(['SD', 'Loc'], [SD, Loc]):
            t0 = time_factor*(pre_time+i*props['episode']+props[episode]['t0_%s'%key])
            tend = t0+time_factor*props[episode]['duration_%s'%key]
            array[(t>t0) & (t<tend)] = props[episode]['amp_%s'%key]

    SD = gaussian_filter1d(SD, int(smoothing/dt))
    Loc = gaussian_filter1d(Loc, int(smoothing/dt))

    props['Nepisode'] = Nep
    props['pre_time'] = pre_time
    props['time_factor'] = time_factor
    return t, SD, Loc

if __name__=='__main__':

    import matplotlib.pylab as plt

    t, SensoryDrive, Locomotion = build_arrays(props)
                                              
    fig3, AX = plt.subplots(2, 1, figsize=(15,2))
    plt.subplots_adjust(bottom=.5)
    AX[0].plot(t, Locomotion, 'k-')
    AX[1].plot(t, SensoryDrive, 'k-')
    for ax, label, key in zip(AX, ['Locomotion', 'Sensory-Drive'], ['SD', 'Loc']):
        ax.axis('off')
        ax.set_xlim([t[0],t[-1]])
        ax.annotate(label+' ', (0,0), ha='right')
    for i, episode in enumerate(list(props.keys())[1:props['Nepisode']+1]):
        t0 = props['time_factor']*(props['pre_time']+i*props['episode']+props['episode']/2.)
        AX[1].annotate('\n'+episode, (t0, 0), 
                       xycoords='data', va='top', ha='center')
    plt.show()

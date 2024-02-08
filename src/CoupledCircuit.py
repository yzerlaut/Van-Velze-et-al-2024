import itertools
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

REC_POPS = ['ThalExc', 'RecExc', 'RecInh', 'VipInh', 'SstInh']
AFF_POPS = ['BgExc', 'ExcToVip', 'ExcToThal']
COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange']

Model = {}
from RecCircuit import Model as Model1
for key in Model1:
    Model[key] = Model1[key]
from DsnhCircuit import Model as Model2
for key in Model2:
    Model[key] = Model2[key]

# Coupling:

Model['p_RecExc_SstInh'] = 0.05
Model['p_RecInh_SstInh'] = 0.05
Model['p_SstInh_RecInh'] = 0.025
Model['p_SstInh_RecExc'] = 0.025

for pre, post in itertools.product(AFF_POPS+REC_POPS, REC_POPS):
    if pre=='ThalExc':
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif (pre=='BgExc') and ('Rec' in post):
        Model['Q_%s_%s'%(pre, post)] = 4. # nS
    elif 'Exc' in pre:
        Model['Q_%s_%s'%(pre, post)] = 2. # nS
    elif 'Inh' in pre:
        Model['Q_%s_%s'%(pre, post)] = 10. # nS



if __name__=='__main__':

    for pre in AFF_POPS+REC_POPS:
        print('----------------------------')
        print(pre, Model['N_%s'%pre])
        for post in REC_POPS:
            if ('p_%s_%s' % (pre, post)) in Model:
                print(pre, post, Model['p_%s_%s' % (pre, post)])
            else:
                print(pre, post, 0)


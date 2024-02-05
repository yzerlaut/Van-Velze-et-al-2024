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

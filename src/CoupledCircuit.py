REC_POPS = ['ThalExc', 'RecExc', 'RecInh', 'VipInh', 'SstInh']
AFF_POPS = ['BgExc', 'AffExc']

Model = {}
from RecCircuit import Model as Model1
for key in Model1:
    Model[key] = Model1[key]
from DsnhCircuit import Model as Model2
for key in Model2:
    Model[key] = Model2[key]

# Coupling:
Model['p_RecExc_SstInh'] = 0.02


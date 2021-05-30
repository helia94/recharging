#pyscript to set all initial files

import os
import json
from charging_prams import *

for name in ['chosen_vehicles','ptv_chosen_vehicles','rejected_by_lack_of_charge']:
    path=path_dic['results']+name+'.csv'
    if os.path.exists(path):
        os.remove(path) 
path=path_dic['auxilary']
for newpath in [path]:
    if not os.path.exists(newpath):
        os.makedirs(newpath)
try:
    open(path+'time_in_simulation.txt', 'w').close()
except FileNotFoundError: pass
with open(path+'time_in_simulation.txt', "w") as f:
    f.write(str(0))
try:
    open(path+'last_used_id_'+'relocation'+'.txt', 'w').close()
except FileNotFoundError: pass
with open(path+'last_used_id_'+'relocation'+'.txt', "w") as f:
    f.write(str(10000))
try:    
    open(path+'last_used_id_'+'charge'+'.txt', 'w').close()
except FileNotFoundError: pass
with open(path+'last_used_id_'+'charge'+'.txt', "w") as f:
    f.write(str(1000000))
    
with open(path+'num_to_charge.json', "w") as f:
        json.dump({'slow':0,'fast':0},f)

for newpath in [path_dic['results'],path_dic['results']+'toure and trajectory/']:
    if not os.path.exists(newpath):
        os.makedirs(newpath)

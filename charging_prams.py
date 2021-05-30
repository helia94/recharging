#-------------main inputs------------------------------------------------------------------
scenario_name='scenario_101_test'
STAGE=3
LOG=1 # to log the requests by charging extension
SoC_in_assignment=True
lazy_charging=False


operation_start_time=6
operation_end_time=22
#------------------------------------------------------------------------------------------




#-------------vehicle properties-----------------------------------------------------------
charging_time=[6,1] # in hr for below 80% SoC, [slow, fast]
DrivingRange=120#km
#------------------------------------------------------------------------------------------




#-------------algorithm steps and reoptimization durations---------------------------------
H=5
time_step_online=240
time_step_B=30*60
#------------------------------------------------------------------------------------------




#-------------objectives and other prameters-----------------------------------------------
LB_SoC_for_trip=20
min_charge_duration={'slow':3, 'fast': 1}

penalty_B={'pick_up':7, 'in_charge_slow':8, 'in_charge_fast':10, 'go_charge_slow':10, 'go_charge_fast':12,'more_in_charge':3}



if SoC_in_assignment:
    penalty_c={'charge':1,'soc_in':0.15,'soc_out':4,'soc_relocate':-4,
                                   'slak_charge_flow_slow':15,'slak_charge_flow_fast':19,'slak_relocation_flow':1,
                                   'slak_cut_charge_flow':1000,'total_in_charge_slow':0.3,'total_in_charge_fast':4,'more_in_charge':3}
else:
    penalty_c={'charge':1,'soc_in':0.15,'soc_out':4,'soc_relocate':-4,
                               'slak_charge_flow_slow':15,'slak_charge_flow_fast':19,'slak_relocation_flow':4,
                               'slak_cut_charge_flow':1000,'total_in_charge_slow':0.3,'total_in_charge_fast':4,'more_in_charge':3}

LB_SoC_to_work=15
R_weights={'cost_to_charger':0.8,'metric':-4,'insertion':1}

#for lazy charging algorithm
stop_charge_limit=90
soc_must_charge=20
#------------------------------------------------------------------------------------------




#-------------do not change----------------------------------------------------------------
files_to_track_performance={'cost_to_charger':{'df':[],
                                     'columns':['time','VehicleId','cost_charge','cost_time','cost_distance','charger','SoC']},
                    'cost_relocation':{'df':[],
                                     'columns':['time','VehicleId','cost_charge','cost_time','cost_distance','from_zone','to_zone','SoC']},
                    'deleted_relocation':{'df':[],
                                     'columns':['time','VehicleId']},
                    'cost_from_charger':{'df':[],
                                     'columns':['time','VehicleId','cost_charge','cost_time','cost_distance','charger','to_zone','SoC']},
                    'SoC_distribution':{'df':[],
                                     'columns':['time','VehicleId','SoC']},
                    'charger_occupation':{'df':[],
                                     'columns':['time','charger','num']},
                    'B_slack_going':{'df':[],
                                     'columns':['time','slow_in','fast_in','slow_go','fast_go', 'X_slow', 'X_fast', 'chargable']},
                    'C_slack':{'df':[],
                                     'columns':['time','flow_slack','in_charge_slow','in_charge_fast']},
                    'charge_gained':{'df':[],
                                     'columns':['time','slow','fast']}}



path_dic={'from_simulator':'',
      'scenario':scenario_name+'/',
      'auxilary':'auxilary/'}
##if lazy_charging:
##    path_dic['results']=path_dic['scenario']+'results/'+'lazy_charging'+'/'
##else:
path_dic['results']=path_dic['scenario']+'results/'+str(STAGE)+'/'
    

charging_rate=[100/(3600*charging_time[0]),100/(3600*charging_time[0]*2),
               100/(3600*charging_time[1]),100/(3600*charging_time[1]*2)]
number_of_steps_in_day=(operation_end_time-operation_start_time+1)*2
assert H> min_charge_duration['slow'], 'make the horizon longer'
if STAGE!=3:
    SoC_in_assignment=False
#------------------------------------------------------------------------------------------

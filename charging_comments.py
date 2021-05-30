
def metric_soc_CG_t():
				  
	This calculates mattrix required to calculate the value of metric used in algorithm R in the report (section 3.4.2). 
	It gets from input the online information on SoC, and information on demand and number vehicles going to charge in rest of the day from algorithm A, which issaved as file after daily planning is run.
	The for is required to take the maximum over all time-slots (equation 3.83).

        
        

def record_tours():
	saves the current tour of veihcles to csv file



def stage_B__write_sparse_matrix():
	
	only executed the first time the charging.py is called
	it constructs some intermidiate matrixs used to make constraints in algorithm B and write them to file as sparse matrixs
	
	the order of original decision variables matrix (X in AX<=b) is assumed as followes:

	first ordered by step h (e.g. 1 2 3), then ordered by origin zone (e.g. a b c) and then by destination zone (e.g. A B C)
	
	with 2 zones and e steps will look like as follows [aA1 aA2  aA3 aB1 aB2 aB3 bA1 bA2  bA3 bB1 bB2 bB3 ]
	


    make_acc_over_zone= it will sum over h steps for each zone
    make_acc_over_zone_by_zone=it will sum over h steps for each flow between the zones
    
    
    reorder_last_by_h_zones=changes the order of variables corrwponding to zones from ordered first by h to last by h 
    reorder_last_by_h_zones_by_zones=changes the order of variables corrwponding to flows between zones from ordered first by h to last by h 
        

    out_from_zone=sum flows going out of each normal zone
    in_to_zone=sum flows going into each normal zone, this is has delay in the steps. meaing when vehicle leave one zone they arrive on estep later to another zone. In equeation 3.59 second sigma is over k=1 to t-1. This is taken care of here.
    out_from_charger=sum flows going into each charger zone
    in_to_charger_for_X=in_to_charger=sum flows going into each charger zone, with no delay in timestep.
    in_to_charger_delayed=np.zeros([number_of_zones_with_charger*h,num_main_vars]), with delay in timestep equal to min_charge_duration. Equations 3,68 and 3.69.
 
    
    zone_to_charger_t0= number of vehicles going out from each zones to chargers only in first step
    zone_to_zone_t0=number of vehicles going out from each zones to other zones only in first step

    sum_over_chargers= sum variables over charger zones

	
def stage_B__get_A_b():

	first it makes the problem with main decison varoables which are flows between the zones, ans then adds all skacl variables. This should be changed. I started with no slack variabels and at the end every day I was adding some. Thefore they are handled well. 
	
    A_ub_dictionary= A in Ax<=b
    b_ub_dictionary= b in Ax<=b
    A_eq_dictionary= A in Ax=b
    b_eq_dictionary= b in Ax=b

    #constraint 1    #num vehicles in zones posetive #@h equation 3.59 (excluding the slack variables)
    A_ub_dictionary['num_vehicles_in_zones']=multiply 3 matrix to get the following function (3. sum over zones posetive for flows going in and negative for flow coming out(2. reshape the matrix to be orderderd by h (1. for each time step, sum the flow over the horizon from the begining, making them acculumative)))
    
	b_ub_dictionary['num_vehicles_in_zones']=number of vehicles in zones in t=0 + (3. reshape the matrix to be orderderd by h(2. for each time step, sum the flow over the horizon from the begining, making them acculumative(1. drop offs with one timestep delay - pickups from the zone)))

    
    #constraint 2    #num vehicles in chargers posetive #@h equation 3.60


    A_ub_dictionary['num_vehicles_in_charger_zones']=multiply 3 matrix to get the following function (3. sum over zones posetive for flows going in and negative for flow coming out(2. reshape the matrix to be orderderd by h (1. for each time step, sum the flow over the horizon from the begining, making them acculumative)))
	
    b_ub_dictionary['num_vehicles_in_charger_zones']=number of vehicles in charger zones in t=0
 
    
    #constraint 3    #chargable vehicles #@t0 equation 3.61
    sparse_A_ub=sparse_matrixs['sparse_zone_to_charger_t0'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    A_ub_dictionary['chargable_0']=multiply 3 matrix to get the following function (3. for each zone, sum the flows going to chager zones in t=0 (1. reshape the matrix to be orderderd by h ))
    b_ub_dictionary['chargable_0']=number of vehicles in each zone that can be sent to charge

    
    #constraint 4    #available vehicles for relocation #@t0 equation 3.62
    sparse_A_ub=sparse_matrixs['sparse_zone_to_zone_t0'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    A_ub_dictionary['avilable_0']= for each zone : sum over flows going to other zones at t=0
    b_ub_dictionary['avilable_0']=for each zone : avilable vehicles at t=0 - customers in the zones at t=0   
    
    #constraint 5    #must cut of charging equation 3.67
	-1* is used to impose greater or equal operator
    sparse_A_ub=(-1*sparse_matrixs['sparse_out_from_charger']).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    A_ub_dictionary['must_cut_charging_h']=multiply 3 matrix to get the following function (-1 * (3. sum over flows going out of each charger zone (2. reshape the matrix to be orderderd by h (1. for each time step, sum the flow over the horizon from the begining, making them acculumative)))
    b_ub_dictionary['must_cut_charging_h']=for each charger zone: -1 * (number of vehicles that must leave the charger for all h (orderded by h))

    #constraint 5    #can cut of charging equation 3.68 and 3.69
    sparse_A_ub=csr_matrix(sparse_matrixs['sparse_out_from_charger'].todense()-sparse_matrixs['sparse_in_to_charger_delayed'].todense()).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    b_ub=np.reshape(can_cut_charging_h,[h*number_of_zones_with_charger],'F')
    A_ub_dictionary['can_cut_charging_h']=sparse_A_ub
    b_ub_dictionary['can_cut_charging_h']=b_ub

    #constraint 6    #in charge vehicles equation 3.63 and 3.64
    sparse_A_eq_base=csr_matrix(-1*sparse_matrixs['sparse_out_from_charger'].todense()+
                                                sparse_matrixs['sparse_in_to_charger'].todense()).dot(
                                                sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(
                                                sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    for type_ in ['slow','fast']:
        sparse_A_eq=sparse_matrixs['sparse_sum_over_chargers'][type_].dot(sparse_A_eq_base)
        b_eq=Y_h[type_]-sparse_matrixs['sparse_sum_over_chargers'][type_].dot(v0_chargers_h)
        A_eq_dictionary['in_charge_'+type_]=sparse_A_eq
        b_eq_dictionary['in_charge_'+type_]=b_eq
    
    
    
    
    #constraint 7    #going to charge vehicles  equation 3.65 and 3.66
    sparse_A_ub_base=-1*sparse_matrixs['sparse_in_to_charger_for_X'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    
    for type_ in ['slow','fast']:
        sparse_A_ub=sparse_matrixs['sparse_sum_over_chargers'][type_].dot(sparse_A_ub_base)
        b_ub=-1*X_h[type_]
        A_ub_dictionary['go_to_charge_'+type_]=sparse_A_ub
        b_ub_dictionary['go_to_charge_'+type_]=b_ub
    #del 

    #constraint 8    #on number of chargers equation 3.60
    sparse_A_ub=A_ub_dictionary['num_vehicles_in_charger_zones']*-1
    b_ub=-1*v0_chargers_h+np.tile(np.array(number_chargers_in_zones_with_charger),h)
    A_ub_dictionary['charger_capacity']=sparse_A_ub
    b_ub_dictionary['charger_capacity']=b_ub
    #del 
    
    
    #add slak variables for flow and in charge 
	
	here I have add the slack variables to the right side of original variables. It is not elegant! [original variables, s_{it} , B_slow_{t} , B'_slow_{t}, B_fast_{t}, B'_fast_{t} , G_slow_{t} , G_fast_{t}, slack for number of chargers (this is missing from the report)]
    

    #constraint 1    #num vehicles in zones posetive #@h  
    sum_over_zone_and_t=np.concatenate([np.concatenate((np.tile(np.eye(number_of_zones),[1,i]),
                                                       np.zeros([number_of_zones,number_of_zones*(h-i)])),1) for i in range(h)],0)
    repeat_zones_in_rows=np.concatenate([np.concatenate([np.zeros([number_of_zones,number_of_zones*(i-1)]),
                                                         np.ones([number_of_zones,number_of_zones]),
                                                       np.zeros([number_of_zones,number_of_zones*(h-i)])],1) for i in range(1,h+1)],0)
    A_ub_right_side=-1*sparse_matrixs['sparse_reorder_last_by_h_zones'].dot(sparse_matrixs['sparse_make_acc_over_zone'])



    A_ub_right_side=A_ub_right_side.todense()
    

    A_ub_right_side+=sum_over_zone_and_t.dot(np.multiply(((sparse_matrixs['sparse_reorder_last_by_h_zones'].todense()).dot(
        drop_offs_h_devided_by_P_shifted)),repeat_zones_in_rows)).dot(sparse_matrixs['sparse_reorder_last_by_h_zones'].todense())
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero=csr_matrix(np.zeros([h*number_of_zones,6*h]))
    sparse_zero_3=csr_matrix(np.zeros([h*number_of_zones,number_of_zones_with_charger*h]))
    A_ub_dictionary['num_vehicles_in_zones']=hstack([A_ub_dictionary['num_vehicles_in_zones'],sparse_A_ub_right_side,sparse_zero,sparse_zero_3])
     
    
    #constraint 4    #avilable vehicles #@t0
    get_slak_t0=csr_matrix(np.concatenate([np.eye(number_of_zones),np.zeros([number_of_zones,number_of_zones*(h-1)])],1))
    A_ub_right_side=-1*get_slak_t0.dot(sparse_matrixs['sparse_reorder_last_by_h_zones'])
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero=csr_matrix(np.zeros([number_of_zones,6*h]))
    sparse_zero_3=csr_matrix(np.zeros([number_of_zones,number_of_zones_with_charger*h]))
    A_ub_dictionary['avilable_0']=hstack([A_ub_dictionary['avilable_0'],sparse_A_ub_right_side,sparse_zero,sparse_zero_3])


    #constraint 6    #in charge vehicles
    A_ub_right_side=np.eye(h)
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero_1=csr_matrix(np.zeros([h,h*number_of_zones]))
    sparse_zero_2=csr_matrix(np.zeros([h,h]))
    sparse_zero_3=csr_matrix(np.zeros([h,number_of_zones_with_charger*h]))
    A_eq_dictionary['in_charge_slow']=hstack([A_eq_dictionary['in_charge_slow'], sparse_zero_1,
                                              sparse_A_ub_right_side,-1*sparse_A_ub_right_side, sparse_zero_2, sparse_zero_2,sparse_zero_2,sparse_zero_2, sparse_zero_3])
    A_eq_dictionary['in_charge_fast']=hstack([A_eq_dictionary['in_charge_fast'], sparse_zero_1,
                                              sparse_zero_2, sparse_zero_2, sparse_A_ub_right_side,-1*sparse_A_ub_right_side,sparse_zero_2, sparse_zero_2, sparse_zero_3])

    #constraint 7    #going to charge vehicles 
    A_ub_right_side=np.zeros([h,h])
##    A_ub_right_side[:-1,1:]=-1*np.eye(h-1)
    A_ub_right_side=-1*np.eye(h)
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero_1=csr_matrix(np.zeros([h,h*number_of_zones]))
    sparse_zero_2=csr_matrix(np.zeros([h,h]))
    sparse_zero_3=csr_matrix(np.zeros([h,number_of_zones_with_charger*h]))
    A_ub_dictionary['go_to_charge_slow']=hstack([A_ub_dictionary['go_to_charge_slow'], sparse_zero_1,
                                               sparse_zero_2, sparse_zero_2,sparse_zero_2,sparse_zero_2,sparse_A_ub_right_side, sparse_zero_2, sparse_zero_3])
    A_ub_dictionary['go_to_charge_fast']=hstack([A_ub_dictionary['go_to_charge_fast'], sparse_zero_1,
                                              sparse_zero_2,sparse_zero_2, sparse_zero_2,sparse_zero_2,sparse_zero_2,sparse_A_ub_right_side, sparse_zero_3])

    
    
    #constraint 8    #on number of chargers
    A_ub_right_side=-1*np.eye(number_of_zones_with_charger*h)
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero_1=csr_matrix(np.zeros([number_of_zones_with_charger*h,h*number_of_zones]))
    sparse_zero_2=csr_matrix(np.zeros([number_of_zones_with_charger*h,6*h]))
    A_ub_dictionary['charger_capacity']=hstack([A_ub_dictionary['charger_capacity'], sparse_zero_1, sparse_zero_2, sparse_A_ub_right_side])

    
    #constraint 9    #pick up slacks should be lower than pickups
    A_ub_right_side=np.eye(np.shape(pick_ups)[0])
    sparse_A_ub_right_side=csr_matrix(A_ub_right_side)
    sparse_zero_2=csr_matrix(np.zeros([h*number_of_zones,6*h]))
    sparse_zero_3=csr_matrix(np.zeros([h*number_of_zones,number_of_zones_with_charger*h]))
    sparse_zero_1=csr_matrix(np.zeros([number_of_zones*h,num_main_vars]))
    A_ub_dictionary['bound_on_pickup_slack']=hstack([sparse_zero_1,sparse_A_ub_right_side,sparse_zero_2,sparse_zero_3])
    b_ub=pick_ups
    b_ub_dictionary['bound_on_pickup_slack']=b_ub
    
 

 
	make C the objective function (Minimize Cx)
    c= np.concatenate([np.reshape(cost[:number_of_zones,:number_of_zones],[num_zone_zone]),
                       np.reshape(cost[:number_of_zones,number_of_zones:],[num_zone_charger]),
                       np.reshape(cost[number_of_zones:,:number_of_zones],[num_zone_charger])])
    c=np.repeat(c,h)
    c_slack_pickup=penalty['pick_up']*np.ones([h*number_of_zones])
    c_slak_charge={}
    c_surplus_charge={}
    c_slack_go_to_charge={}
    for type_ in ['slow','fast']:
        c_slak_charge[type_]=penalty['in_charge_'+type_]*np.ones([h])
        c_slack_go_to_charge[type_]=penalty['go_charge_'+type_]*np.ones([h])
        c_surplus_charge[type_]=penalty['more_in_charge']*np.ones([h])
    extra_charger_capacity=1000*np.ones([number_of_zones_with_charger*h])
        
    c=np.concatenate((c,c_slack_pickup,c_slak_charge['slow'],c_surplus_charge['slow'],c_slak_charge['fast'],c_surplus_charge['fast'],c_slack_go_to_charge['slow'],c_slack_go_to_charge['fast'],extra_charger_capacity),0)



def stage_B__solve_gurobi():

    write the matrix Ax<=b in gurobi format	
	sets gap and time limit for stopping the branch and bound



    

class XServer:   
	for travel time requests
 

    
def link_consumption ():
    to calculate battery consumption



def read_simulation_files():
    def get_futur_locations():
		gets planned tours of vehicles from files written by simulator


    def get_past_locations():
        reads privious trejectories since the last call 





if not lazy_charging:
    def make_fleet_attribute():
		
		make file that trackes battery/ charging status of all vehicles in panda 



    def make_files_to_track_performance():
		a function to make and update files for results

    def read_static_data():
		funcion that reads basic information that do not change in iterations, location od chargers, etc.
		DailyPlan comes from algorithm A which is in the jupyter notebook



    def update_fleet_attribute_with_simulation_data(): 
		manages the changes to SoC and status of vehicles in charge (if they are on the way to charge, if in charge). 
		Check in which zone their final destination is
		Check if they ahve enough charge for a trip
		Check if their charge is low enough to be candidate for sending to charge
       

    def write_tours_and_soc_for_R():
		writes down tours of vehicles for options.py

        
    def get_B_input():
		aggreates information over zone to be used in algorithm B. I did everything in panda and after all calculation I convert to matrix.
        count_for_B_vehicles= for number fo vehicles currently in zones
        count_for_B_chargers=for number fo vehicles currently in charger zones
       
        pick_ups_h=number of expected pickups for h steps
        drop_offs_h=number of expected dropoffs for h steps

        df_cost_matrix=cost from origin zone to destination zone as panda dataframe

        cost=cost from origin zone to destination zone as matrix

        can_cut_charging_h=number fo vehicles in charger zones that CAN stop charging during h steps
        must_cut_charging_h=number fo vehicles in charger zones that MUST stop charging during h steps


        avilable_0= number of vehicles that have enough battery for a trip in zones
        chargable_0=number of vehicles that can go to charge in zones
        X_h=number of vehicles that should go to charge in h steps (slow and fast)
        Y_h=number of vehicles that should be in charge in h steps (slow and fast)
        
       

    def get_C_input():
		manages data needed for algorithm C
        def fromat_B_answer():
			
			from all decision variables in algorithm B it extracrs posetive flows in the first step h=1
            

            df_B_set_1=flows from zones to chargers
            df_B_set_2=relocation flows (from zones to zones)
            df_B_set_3=flows from chargers to zones
            B_in_charge=Number of vehcile that should be in charge in h=1 according to algorithm B

        def get_potential_vehicles():
			in origin zones that had a posetive flow, find potential vehciles 
        def get_cost_for_options():
            for all possible vehicles-destination pairs request distance and traveltime from xroute. 
			distance for flows from chargers to zone centers are calculated offline
			def get_charge_in_t_min()
				for vehicles that we are considering to stop their charging, calculate what their SoC will be in 7.5, 15, 22.5, and 30 minutes



        def get_charger_state():
            check how many vehicles are in charge anon the way to charge for each charger





    def C__get_A_ub ():
		construct A and b in Ax<=b fro algorithm C.
		It is messy
		The code is not optimized takes actually quite a lot because back and forth between numpy and pandas, and converting between integer index to boelian index.
		Basicly you have to match variables to vehicles, to slacks and flows between zones. 
        
		code "set_id" for type of flows from algorithm B
		relocation 2
		going to slow charger 11
		going to fast charger 12
		coming back from slow charger 31
		coming back from fast charger 32
 
		variables x in Ax<=b are assumed in this order [vehicle-destination assignment , slak_for_flows, total_in_charge_slaks_slow, 
										total_in_charge_slaks_fast, more_in_charge_slow, more_in_charge_fast]
 
            #first constraint, flow from B should be satisfied 
			there is one constraint for each origin-destination pair 
			
			get avilable vehicles from df_with_vehicles
			get flows of algorithm B from df_with_B_flows

			

            #charger capacity should not exceed  for each t in [0,1,2,3], corresponding to [0, 7.5, 15, 22.5] minutes from now
			
			get charger capacity and number of cars in the chargers from chargers_state and aggregate to get the number to eahc station
			get cars already on their way to charger from df_planed_charging_t and aggregate to get the number to eahc station
			number of constraint would be number of stations that have either cars in them or cars to eb sent towards them
			


            #each vehicle can only fulfill one destination
            df_get_matching_indexs=df_with_vehicles.groupby(['VehicleId'])['vehicle_action_pair_var_id'].apply(list).reset_index()
            df_get_matching_indexs['vehicle_action_pair_var_id']=df_get_matching_indexs['vehicle_action_pair_var_id'].apply(
                                                                                                                    lambda x: np.array(x)).apply(
                                                                                                                    lambda x: make_range('original',
                                                                                                                    x, dic_length_of_var_types))

            df_get_matching_indexs['indexs_with_value_1_for_costraint_on_vehicles']=df_get_matching_indexs['vehicle_action_pair_var_id']
            df_get_matching_indexs['indexs_with_value_1_for_costraint_on_vehicles']=df_get_matching_indexs['indexs_with_value_1_for_costraint_on_vehicles'].apply(make_boelian_index)
            indexes_for_A_ub = np.array(df_get_matching_indexs['indexs_with_value_1_for_costraint_on_vehicles'].tolist())

            A_ub=make_x(len(df_get_matching_indexs))
            A_ub[indexes_for_A_ub]=1
            b_ub=np.ones([len(indexes_for_A_ub)])
            dictionaty_A_ub['car_occupied']=csr_matrix(A_ub)
            dictionaty_b_ub['car_occupied']=b_ub   


            #on total_in_charge for each t 
            for type_ in ['slow','fast']:
                for t in range(4):
                    df_set_1=df_with_vehicles[(df_with_vehicles['set_id']==set_id_dic['in'][type_])]
                    df_set_3=df_with_vehicles[(df_with_vehicles['set_id']==set_id_dic['out'][type_])]


                    df_set_1=df_set_1[df_set_1['EAT']+df_set_1['cost_time']<= (t+1)*time_bound_step]
                    df_set_3=df_set_3[df_set_3['t']<=t]

                    df_planed_charging_t=df_planed_charging[(df_planed_charging['EAT']<=(1+t)*time_bound_step)&(df_planed_charging['charger'].isin(list_of_chargers[type_]))]
                    in_charge_preload=df_planed_charging_t['charging_planed_'+type_].sum()+chargers_state['in_charge_'+type_].sum()
                    index_in=df_set_1['vehicle_action_pair_var_id'].values
                    index_out=df_set_3['vehicle_action_pair_var_id'].values

                    index_in=make_range('original', index_in, dic_length_of_var_types)
                    index_out=make_range('original', index_out, dic_length_of_var_types)

                    A_ub=make_x(1)
                    A_ub[:,index_in]=-1
                    A_ub[:,index_out]=1
                    index_slak=make_range('total_in_charge_slaks_'+type_, np.array([t]), dic_length_of_var_types)
                    A_ub[:,index_slak]=-1
                    index_slak=make_range('more_in_charge_'+type_, np.array([t]), dic_length_of_var_types)
                    A_ub[:,index_slak]=+1
                    b_ub=-1*np.array([B_in_charge[type_]-in_charge_preload])
                    dictionaty_A_eq['satisfy_total_in_charge_'+type_+'_'+str(t)]=csr_matrix(A_ub)
                    dictionaty_b_eq['satisfy_total_in_charge_'+type_+'_'+str(t)]=b_ub    





            #objective function  
            def penalty_SoC_to_charg(soc):
                penalty=objective_weigths['soc_in']*soc
                return penalty
            A_LB=DailyPlan['mean_charge_cut_charge'][time_step]-15
            A_UB=DailyPlan['mean_charge_cut_charge'][time_step]+15
            A_UB=min(A_UB,95)
            if np.isnan(A_LB):
                A_LB=40
            if np.isnan(A_UB):
                A_UB=95
            def penalty_SoC_cut_charg(soc,A_LB=A_LB,A_UB=A_UB,penalty_L=0.02,penalty_H=0.01):
                if soc<=A_UB and soc>=A_LB:
                    penalty=0 
                    return penalty
                elif soc>A_UB:
                    penalty=(soc-A_UB)*penalty_H
                    return penalty
                elif soc<A_LB:
                    penalty=(-soc+A_LB)*penalty_L
                    return penalty

                
            if STAGE==3 and SoC_in_assignment:    
                path=path_dic['auxilary']
                with open(path+'soc_metric_t.json', "r") as f:
                    soc_metric_t=json.load(f)
                    
                def get_soc_metric_relocate(soc):
                    metric_list=[]
                    for above in list(soc_metric_t.keys()):
                        if soc>=int(above):
                            metric_list+=[soc_metric_t[above]]
                    if len(metric_list)>0:
                        metric=max(metric_list)
                    else:
                        metric=1
                    return metric 

            c=make_x(1)
            c=c[0,:]
            df_objective_vehicles=df_with_vehicles
            df_objective_vehicles['c']=0
            df_objective_vehicles['c']+=objective_weigths['charge']*df_objective_vehicles['cost_charge']
            df_objective_vehicles.loc[df_objective_vehicles['set_id'].isin([11,12]),'c']+=df_objective_vehicles[df_objective_vehicles['set_id'].isin([11,12])]['SoC'].apply(
                                                                                                    lambda x: penalty_SoC_to_charg(x)).values


            #gettting ranking for vehicles to stop charging
            df_rank=df_objective_vehicles.loc[df_objective_vehicles['set_id'].isin([31,32])][['VehicleId','vehicle_action_pair_var_id','origin','destination','SoC','t']]
            
            df_rank_0=df_rank[df_rank['t']==0][['VehicleId','SoC']].drop_duplicates()
            df_rank_0['rank_by_veh']=df_rank_0['SoC'].rank(ascending=False)
            df_rank=pd.merge(df_rank,df_rank_0[['VehicleId','rank_by_veh']],how='left',on='VehicleId')
            df_rank['rank_by_t']=df_rank['SoC'].apply(penalty_SoC_cut_charg)
            df_rank['rank_over_all']=df_rank['rank_by_veh']*10+df_rank['rank_by_t']
            df_rank['rank_over_all']=df_rank['rank_over_all'].rank(pct=True)
            df_rank['rank_over_flow_group']= df_rank.groupby(['origin','destination'])['rank_over_all'].rank()
            
            df_objective_vehicles.loc[df_objective_vehicles['set_id'].isin([31,32]),'c']+=df_rank['rank_over_flow_group'].values*objective_weigths['soc_out']
            if STAGE==3 and SoC_in_assignment:
                df_objective_vehicles['soc_metric_relocate']=0            
                df_objective_vehicles.loc[df_objective_vehicles['set_id'].isin([2]),'soc_metric_relocate']=df_objective_vehicles[df_objective_vehicles['set_id'].isin([2])]['SoC'].apply(lambda x: get_soc_metric_relocate(x)).values
                df_objective_vehicles.loc[df_objective_vehicles['set_id'].isin([2]),'c']+=objective_weigths['soc_relocate']*df_objective_vehicles[df_objective_vehicles['set_id'].isin([2])]['soc_metric_relocate'].values

            indexs=make_range('original', df_objective_vehicles['vehicle_action_pair_var_id'].values , dic_length_of_var_types)
            c[indexs]=df_objective_vehicles['c'].values   

            for type_ in ['slow','fast']:
                df_objective_flow_slacks=df_with_B_flows[(df_with_B_flows['set_id']==set_id_dic['in'][type_])]
                df_objective_flow_slacks['c']=objective_weigths['slak_charge_flow_'+type_]
                indexs=make_range('slak_for_flows', df_objective_flow_slacks['flow_pair_var_id'].values, dic_length_of_var_types)
                c[indexs]=df_objective_flow_slacks['c'].values   


            df_objective_flow_slacks=df_with_B_flows[df_with_B_flows['set_id']==2]
            df_objective_flow_slacks['c']=objective_weigths['slak_relocation_flow']
            indexs=make_range('slak_for_flows', df_objective_flow_slacks['flow_pair_var_id'].values , dic_length_of_var_types)
            c[indexs]=df_objective_flow_slacks['c'].values   

            df_objective_flow_slacks=df_with_B_flows[df_with_B_flows['set_id'].isin([31,32])]
            df_objective_flow_slacks['c']=objective_weigths['slak_cut_charge_flow']
            indexs=make_range('slak_for_flows', df_objective_flow_slacks['flow_pair_var_id'].values , dic_length_of_var_types)
            c[indexs]=df_objective_flow_slacks['c'].values  
       

            indexs=make_range('total_in_charge_slaks_slow', np.arange(4), dic_length_of_var_types)
            c[indexs]=objective_weigths['total_in_charge_slow']

            indexs=make_range('total_in_charge_slaks_fast', np.arange(4), dic_length_of_var_types)
            c[indexs]=objective_weigths['total_in_charge_fast']


            indexs=make_range('more_in_charge_slow', np.arange(4), dic_length_of_var_types)
            c[indexs]=objective_weigths['more_in_charge']
            indexs=make_range('more_in_charge_fast', np.arange(4), dic_length_of_var_types)
            c[indexs]=objective_weigths['more_in_charge']

            A_ub=vstack(list(dictionaty_A_ub.values()))
            b_ub=np.concatenate(list(dictionaty_b_ub.values()),0)

            A_eq=vstack(list(dictionaty_A_eq.values()))
            b_eq=np.concatenate(list(dictionaty_b_eq.values()),0)


            c=c.flatten()
            S_ub = coo_matrix(A_ub).tocsr()
            S_eq = coo_matrix(A_eq).tocsr()

            return [S_ub, b_ub, S_eq, b_eq, c, dictionaty_A_ub, dictionaty_b_ub, dictionaty_A_eq, dictionaty_b_eq, dic_length_of_var_types,df_with_vehicles], 1
        else:
            return [],0


    def C__solve_A_ub ():
        feed the problem to gurobi



    def apply_actions(C_answer, df_with_vehicles,files_to_track_performance,fleet_attribute,dic_length_of_var_types, code=1):
        def get_posetive_actions_form_c (C_answer, df_with_vehicles,files_to_track_performance):

            C_answer=np.rint(C_answer)
            s_1=C_answer[dic_length_of_var_types['original']:+dic_length_of_var_types['original']+dic_length_of_var_types['slak_for_flows']]

            s_2=C_answer[dic_length_of_var_types['original']+dic_length_of_var_types['slak_for_flows']:dic_length_of_var_types['original']+dic_length_of_var_types['slak_for_flows']+dic_length_of_var_types['total_in_charge_slaks_slow']]

            s_3=C_answer[dic_length_of_var_types['original']+dic_length_of_var_types['slak_for_flows']+dic_length_of_var_types['total_in_charge_slaks_slow']:]

      
            df_with_vehicles['answer']=C_answer[:len(df_with_vehicles)]
            df_with_vehicles=df_with_vehicles[df_with_vehicles['answer']==1]


            
            df_with_vehicles=df_with_vehicles[((df_with_vehicles['set_id'].isin([31,32]))&(df_with_vehicles['t']==0))|
                                             ((df_with_vehicles['set_id'].isin([11,12]))&(df_with_vehicles['EAT']<2*time_step_online))|
                                             ((df_with_vehicles['set_id'].isin([2]))&(df_with_vehicles['EAT']<time_step_online))]

            df_set_1=df_with_vehicles.loc[df_with_vehicles['set_id'].isin([11,12]),['VehicleId','charger','cost_time']]
            df_set_2_3=df_with_vehicles.loc[df_with_vehicles['set_id'].isin([2,31,32]),['VehicleId','destination','cost_time']]

            df_set_1=pd.merge(df_set_1,charger_attribute[['charger_id', 'charger_location']],how='left',left_on='charger',right_on='charger_id')
            df_set_2_3=pd.merge(df_set_2_3,zone_attribute[['zone','zone_center_coord']],how='left',left_on='destination',right_on='zone')

        def make_request_column(df_set_1,df_set_2_3):

            def make_charge_relocate_request(TripId, coord, VehicleId, time, time_now):
                request={"TripId":int(TripId),
                         "Time":time_now,
                         "TargetServiceTime":time,
                         "TargetLocation":{"X":coord[0],"Y":coord[1]},
                         "VehicleId":int(VehicleId)}
                return request


            last_used_id={}
            time_now={"Seconds":time_in_simulation,"Nanos":0}
            duration={"Seconds":(operation_end_time-operation_start_time)*3600-time_in_simulation,"Nanos":0}
            path=path_dic['auxilary']
            for id_ in ['charge','relocation']:
                with open(path+'last_used_id_'+id_+'.txt', "r") as f:
                    last_used_id[id_]=int(f.read())
                open(path+'last_used_id_'+id_+'.txt', 'w').close()

            df_set_1.index = range(1,len(df_set_1.index)+1)
            df_set_1['id']=(df_set_1.index+last_used_id['charge']).astype(int)
            last_used_id['charge']+=len(df_set_1.index)


            if len(df_set_1)>0:
                df_set_1['json_format']=df_set_1.apply(
                                            lambda row: make_charge_relocate_request(row['id'], row['charger_location'],
                                                                                     row['VehicleId'],duration,time_now), axis=1)
                path=path_dic['auxilary']
                with open(path+'num_to_charge.json', "r") as f:
                    num_to_charge=json.load(f)
                    a=0
                    for type_ in['slow','fast']:
                        num_to_charge[type_]+=len(df_set_1[df_set_1['charger'].isin(list_of_chargers[type_])])
                        a+=len(df_set_1[df_set_1['charger'].isin(list_of_chargers[type_])])
                    assert a==len(df_set_1),'not all chargers in list'
                with open(path+'num_to_charge.json', "w") as f:
                    json.dump(num_to_charge,f)
                
                if LOG==1:
                    print('send to charge', df_set_1['VehicleId'].tolist())
            else:
                df_set_1['json_format']=0

            df_set_2_3.index = range(1,len(df_set_2_3.index)+1)
            df_set_2_3['id']=(df_set_2_3.index+last_used_id['relocation']).astype(int)
            last_used_id['relocation']+=len(df_set_2_3.index)

            duration={"Seconds":0,"Nanos":0}
            if len(df_set_2_3):
                df_set_2_3['json_format']=df_set_2_3.apply(
                                            lambda row: make_charge_relocate_request(row['id'], row['zone_center_coord'],
                                                                                     row['VehicleId'],duration,time_now), axis=1)
            if LOG==1:
                if len( df_set_2_3['VehicleId'].tolist())>0:
                    print('relocate VehicleId', df_set_2_3['VehicleId'].tolist())

            path=path_dic['auxilary']
            for id_ in ['charge','relocation']:    
                with open(path+'last_used_id_'+id_+'.txt', "w") as f:
                    f.write(str(last_used_id[id_]))
            return df_set_1,df_set_2_3

        def make_delete_request_column(fleet_attribute):
            #delete old request if vehicle has new action
            time_now={"Seconds":time_in_simulation,"Nanos":0}
            def make_delete_request(TripId, VehicleId,time_now):
                request={"Time":time_now,
                         "TripId":int(TripId) ,
                         "VehicleId":int(VehicleId)}
                return request
            list_of_vehicles_with_new_actions=df_set_1['VehicleId'].tolist()+df_set_2_3['VehicleId'].tolist()
            df_delete_request=fleet_attribute.loc[(fleet_attribute['VehicleId'].isin(list_of_vehicles_with_new_actions))&
                               (fleet_attribute['trip_id']>0)]
            list_VehicleId_delete1=df_delete_request['VehicleId'].tolist()
            df_delete_request=df_delete_request[['VehicleId','trip_id']]
            if LOG==1:
                if len(df_delete_request['VehicleId'].tolist())>0:
                    print('delete charging request_VehicleId', df_delete_request['VehicleId'].tolist())
            if len(df_delete_request)>0:
                df_delete_request['json_format']=df_delete_request.apply(
                                            lambda row: make_delete_request(row['trip_id'],
                                                                                     row['VehicleId'],time_now), axis=1)

            #delete relocation if vehicle has customer    
            list_of_vehicles_with_planned_relocation=fleet_attribute.loc[fleet_attribute['relocation_planed']>0,
                                                                   'VehicleId'].tolist()
            list_of_vehicles_with_planned_relocation_that_got_customer=df_future.loc[df_future['VehicleId'].isin
                                                                                     (list_of_vehicles_with_planned_relocation)&
                                    df_future['TripIds'].apply(lambda x:len(x)>1),'VehicleId'].tolist()
            df_delete_request_2=fleet_attribute.loc[fleet_attribute['VehicleId'].isin(
                            list_of_vehicles_with_planned_relocation_that_got_customer)]
            df_delete_request_2=df_delete_request_2[df_delete_request_2['VehicleId'].isin(list_VehicleId_delete1)==0]
            df_delete_request_2=df_delete_request_2[['VehicleId','trip_id']]
    ##        if LOG==1:
    ##            print('delete relocating vehicle if got passenger_VehicleId', df_delete_request_2['VehicleId'].tolist())
    ##            print('delete relocating vehicle if got passenger_TripId', df_delete_request_2['trip_id'].tolist())
            if len(df_delete_request_2)>0:
                df_delete_request_2['json_format']=df_delete_request_2.apply(
                                            lambda row: make_delete_request(row['trip_id'],
                                                                                     row['VehicleId'],time_now), axis=1)
            performance_update={}
            performance_update['deleted_relocation']=df_delete_request_2[['VehicleId']]
            return df_delete_request, df_delete_request_2



        def write_requests_to_file():
            path=path_dic['from_simulator']
            with open(path+'EVehicleChargingRequests.json', "w") as f:
                request_list=[]
                if len(df_set_1)>0:
                    request_list+=df_set_1['json_format'].tolist()
                if len(df_set_2_3)>0:
                    request_list+=df_set_2_3['json_format'].tolist()
                json.dump(request_list, f, default=str)

            with open(path+'EVehicleChargingDeletionRequests.json', "w") as f:
                request_list=[]
                if len(df_delete_request)>0:
                    request_list+=df_delete_request['json_format'].tolist()
    ##            if len(df_delete_request_2)>0:
    ##                request_list+=df_delete_request_2['json_format'].tolist()
                json.dump(request_list, f, default=str)



        def update_fleet_attribute(fleet_attribute,df_set_1,df_set_2_3):
            # delete old plans

            list_to_delete=df_delete_request['VehicleId'].tolist()#+df_delete_request_2['VehicleId'].tolist()
            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_delete),
                                ['charging_planed_slow',
                                 'charging_planed_fast',
                                 'Expected_start_charging_time',
                                 'Expected_arrival_on_relocation',
                                 'in_charge_slow',
                                 'in_charge_fast',
                                 'relocation_planed',
                                 'charger',
                                 'charge_duration',
                                'trip_id']]=0
            
                                            
            #write in new plans
            df_set_1_=pd.merge(df_set_1,charger_attribute[['charger_id','fast','slow']],how='left',
                              left_on='charger',right_on='charger_id')
            df_set_1_=df_set_1_[['VehicleId','charger','fast','slow','id','cost_time']]

            df_set_1_.sort_values(by=['VehicleId'],inplace=True)
            fleet_attribute.sort_values(by=['VehicleId'],inplace=True)

            list_to_charge_slow=df_set_1_.loc[df_set_1_['slow']==1,'VehicleId'].tolist()
            list_to_charge_fast=df_set_1_.loc[df_set_1_['fast']==1,'VehicleId'].tolist()
            list_to_charge=list_to_charge_slow+list_to_charge_fast

            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_charge_slow),
                               'charging_planed_slow']=1
            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_charge_fast),
                               'charging_planed_fast']=1


            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_charge),
                               'charger']=df_set_1_['charger'].values

            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_charge),
                               'trip_id']=df_set_1_['id'].values

            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_charge),
                               'Expected_start_charging_time']=fleet_attribute.loc[
                                fleet_attribute['VehicleId'].isin(list_to_charge),
                               'EAT'].values+df_set_1_['cost_time'].values+time_in_simulation
            #for relocation

            df_set_2_3=df_set_2_3[['VehicleId','id','cost_time']]

            df_set_2_3.sort_values(by=['VehicleId'],inplace=True)
            fleet_attribute.sort_values(by=['VehicleId'],inplace=True)

            list_to_relocate=df_set_2_3['VehicleId'].tolist()

            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'relocation_planed']=1
            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'in_charge_slow']=0
            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'in_charge_fast']=0
            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'trip_id']=df_set_2_3['id'].values

            fleet_attribute.loc[fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'Expected_arrival_on_relocation']=fleet_attribute.loc[
                                fleet_attribute['VehicleId'].isin(list_to_relocate),
                               'EAT'].values+df_set_2_3['cost_time'].values+time_in_simulation
            
            return fleet_attribute
        def write_fleet_attribute(fleet_attribute):
            fleet_attribute=fleet_attribute[['VehicleId',
                                           'SoC_now',
                                           'charging_planed_slow',
                                           'charging_planed_fast',
                                           'relocation_planed',
                                           'Expected_start_charging_time',
                                           'Expected_arrival_on_relocation',
                                           'in_charge_slow', 
                                           'in_charge_fast', 
                                           'charger', 
                                           'charge_duration',
                                           'trip_id',
                                             'last_row_trejectory']]
            path=path_dic['auxilary']

            fleet_attribute.to_csv(path+'fleet_attribute.csv',sep=';',index=False)

        if code==1:
            df_set_1, df_set_2_3,files_to_track_performance=get_posetive_actions_form_c(C_answer, df_with_vehicles,files_to_track_performance)
            fleet_attribute[['VehicleId','trip_id']]=fleet_attribute[['VehicleId','trip_id']].astype(int)
            df_set_1[['VehicleId']]=df_set_1[['VehicleId']].astype(int)
            df_set_2_3[['VehicleId']]=df_set_2_3[['VehicleId']].astype(int)
            df_set_1,df_set_2_3=make_request_column(df_set_1,df_set_2_3,)
            df_delete_request, df_delete_request_2=make_delete_request_column(fleet_attribute)
            write_requests_to_file()
            fleet_attribute=update_fleet_attribute(fleet_attribute,df_set_1,df_set_2_3)
        write_fleet_attribute(fleet_attribute) 
        _=make_files_to_track_performance(files_to_track_performance,'write')




    time_in_simulation+=time_step_online
    time_of_day_in_hr=operation_start_time+int(time_in_simulation/(30*60))/2
    time_step=int(time_in_simulation/(30*60))
    continue_script=record_tours()

    if continue_script==1 and STAGE!=1:        
        if time_in_simulation==time_step_online:
            df_past, df_future=read_simulation_files()
            VehicleId_list=df_future['VehicleId'].tolist()
            feet_size=len(VehicleId_list)
            fleet_attribute=make_fleet_attribute(VehicleId_list)
            files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'make')

        else:
            path=path_dic['auxilary']
            fleet_attribute=pd.read_csv(path+'fleet_attribute.csv',sep=';')
            df_past, df_future=read_simulation_files()
            VehicleId_list=df_future['VehicleId'].tolist()
            feet_size=len(VehicleId_list)


        (charger_attribute,charger_zone_attribute,number_chargers_in_zones_with_charger, 
        number_slow_charger_zones,number_fast_charger_zones,number_of_zones_with_charger,
        zone_attribute,number_of_zones,number_of_all_zones,
        cost_charger_to_zone_center, zones, DailyPlan, list_of_chargers)=read_static_data()
		
        steps_in_one_day=len(DailyPlan['X_day']['slow'])
		
        h=max(1,min(H,steps_in_one_day-time_step))
        
        fleet_attribute,files_to_track_performance=update_fleet_attribute_with_simulation_data(fleet_attribute,files_to_track_performance)

        write_tours_and_soc_for_R()

        
        initial_v0,pick_ups_h,drop_offs_h,cost,must_cut_charging_h, can_cut_charging_h, avilable_0, chargable_0,count_for_B_vehicles, X_h, Y_h=get_B_input()

        if time_in_simulation==time_step_online:
            stage_B__write_sparse_matrix(H, number_of_zones, 
                                 number_slow_charger_zones, number_fast_charger_zones)


        S_ub, b_ub,S_eq, b_eq, c, A_ub_dictionary,A_eq_dictionary, b_ub_dictionary, b_eq_dictionary=stage_B__get_A_b(H, number_of_zones, number_chargers_in_zones_with_charger,
                             number_slow_charger_zones, number_fast_charger_zones, 
                             initial_v0,pick_ups_h,drop_offs_h , 
                             X_h,Y_h,
                             cost,must_cut_charging_h, can_cut_charging_h, 
                             avilable_0, chargable_0,penalty_B)
        B_answer, B_success=stage_B__solve_gurobi(S_ub, b_ub,S_eq, b_eq, c, 20)

        df_set_1, df_set_2, df_set_3,df_planed_charging,df_planed_relocation,chargers_state,B_in_charge=get_C_input(B_answer,files_to_track_performance)

                
        list_from_C,posetive_flow_from_B=C__get_A_ub (df_set_1, df_set_2, df_set_3,
                                                                    list_of_chargers, 
                                                                    df_planed_charging,df_planed_relocation,
                                                                    chargers_state,B_in_charge,UP_SoC_for_charging=50,
                                                                    LB_SoC_for_stop_charging=70,objective_weigths=penalty_c,time_bound_step=(time_step_B/4))

        if posetive_flow_from_B==1:
            [S_ub, b_ub, S_eq, b_eq, c, dictionaty_A_ub, dictionaty_b_ub, dictionaty_A_eq, dictionaty_b_eq, dic_length_of_var_types, df_with_vehicles]=list_from_C
            C_answer, C_success=C__solve_A_ub (S_ub, b_ub, S_eq, b_eq, c, dic_length_of_var_types, gap=0.01)
            
            apply_actions(C_answer, df_with_vehicles,files_to_track_performance,fleet_attribute,dic_length_of_var_types)
        else:
            apply_actions([], [],files_to_track_performance,fleet_attribute,[], code=0)



from __future__ import (absolute_import, division, print_function)
from charging_prams import *

#basic libraries
import os
import glob
import json
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from requests import post
from datetime import datetime, timedelta
import time
import pickle

# to silent warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#formatting data
from  ast import literal_eval
from pandas.io.json import json_normalize
from copy import deepcopy

if STAGE in [2,3]:
    #for shapefiles 
    from shapely.geometry import Point
    from geopandas import GeoSeries, GeoDataFrame,read_file, sjoin
    from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
    from functools import partial

    #calculation                                                                                                                              
    import gurobipy
    from scipy.sparse import csr_matrix, hstack, vstack,coo_matrix, save_npz, load_npz

def metric_soc_CG_t(t,number_of_time_slots,stop_charge,go_charge,mean_charge_cut_charge,mean_charge_in_charge,charge,
                  in_charge,decharge_rate,IP_demand_sum_over_zones,group_size,for_above,charge_go_to_charge_all, charge_cut_charge_all):
    index_array=np.ones([number_of_time_slots,1])
    free_capacity=np.zeros([number_of_time_slots])
    capacity_of_above=np.zeros([number_of_time_slots])
    metric_t=np.zeros([1])
    t_critical=np.zeros([number_of_time_slots,1])
    delta_mat=np.zeros([number_of_time_slots])
    metric_t=np.zeros([1])
    for delta in range (1,min(38,number_of_time_slots-t)):
        tt=t+delta
        t_list=list(range(t,tt+1))
        assert len(t_list)==delta+1

        t_out_of_charge=np.sum(stop_charge,0)[t_list]
        t_go_charge=np.sum(go_charge,0)[t_list]
        t_charge_out_of_charge=mean_charge_cut_charge[[t_list]].astype(float)   
        t_charge_out_of_charge=np.nan_to_num(t_charge_out_of_charge)

        t_charge_in_charge=mean_charge_in_charge[[t_list]].astype(float)   
        t_charge_in_charge=np.nan_to_num(t_charge_in_charge)

        t_num_can_serve_from_chargers=np.minimum(np.round(((1/decharge_rate)*(t_charge_out_of_charge)),0).astype(int),np.array(list(reversed(range(1,len(t_list)+1)))))

        t_num_have_to_go_to_charge=((1/decharge_rate)*(t_charge_in_charge)).astype(int)
        t_num_have_to_go_to_charge=np.repeat(t_num_have_to_go_to_charge,t_go_charge)
        t_num_can_serve_from_chargers[0]=0
        if len(t_num_have_to_go_to_charge)>0:
            t_num_have_to_go_to_charge[0]=0
        t_num_can_serve_from_chargers=np.repeat(t_num_can_serve_from_chargers,t_out_of_charge)

        charge_=deepcopy(charge)
        charge_[in_charge.astype(bool)]=0
        num_can_serve=((1/decharge_rate)*(charge_)).astype(int)
        num_can_serve=np.minimum(num_can_serve,np.ones([len(num_can_serve)])*(delta+1))
        total_num_can_serve=num_can_serve
        total_num_can_serve=np.tile(total_num_can_serve,group_size)
        total_num_can_serve=total_num_can_serve[total_num_can_serve>0]
        total_num_can_serve_sort=total_num_can_serve[np.argsort(total_num_can_serve)]
        t_demand=IP_demand_sum_over_zones[t_list]

        number_of_vehicles_above=np.sum(total_num_can_serve_sort>int(for_above/decharge_rate))


        put_roof_for_can_serve=lambda x: np.minimum(x,np.tile((np.array(list(reversed(range(1,len(t_list)+1))))[np.newaxis,:]),[np.shape(x)[0],1]))
        charge_go_to_charge_all_=np.nan_to_num(charge_go_to_charge_all[:,t_list].astype(float))
        charge_cut_charge_all_=np.nan_to_num(charge_cut_charge_all[:,t_list].astype(float))
        charge_go_to_charge_all_=((1/decharge_rate)*charge_go_to_charge_all_).astype(int)
        charge_cut_charge_all_=((1/decharge_rate)*charge_cut_charge_all_).astype(int)
        charge_go_to_charge_all_[:,0]=0
        charge_cut_charge_all_[:,0]=0
        charge_go_to_charge_all_=put_roof_for_can_serve(charge_go_to_charge_all_)
        charge_cut_charge_all_=put_roof_for_can_serve(charge_cut_charge_all_)            
        index_above_50=np.sum(total_num_can_serve_sort<(for_above/decharge_rate))
        free_capacity[t_list[-1]]=np.sum(total_num_can_serve_sort)+np.sum(charge_cut_charge_all_.flatten())-np.sum(t_demand)-np.sum(charge_go_to_charge_all_.flatten())
        capacity_of_above[t_list[-1]]=np.sum(total_num_can_serve_sort[index_above_50:])
##        if free_capacity[t_list[-1]]<0:
##            print('dont have charge over:', t_list)
        delta_mat[t_list[-1]]=len(t_list)

    return free_capacity, capacity_of_above,delta_mat, number_of_vehicles_above
        
        

def record_tours():
    columns_used=[
         'CurrentLocation.X',
         'CurrentLocation.Y',
         'Tour.NextStopIndex',
         'Tour.Stops',
         'Tour.Trips',
         'VehicleId']  
    if time_in_simulation==time_step_online:
        df_EVehicleTours_day=pd.DataFrame(columns=columns_used+['time'])
        path=path_dic['results']+'toure and trajectory/'
        df_EVehicleTours_day.to_csv(path+'df_EVehicleTours_day.csv',index=False,sep=';')
    path=path_dic['from_simulator']
    with open(path+'EVehicleTours.json') as f:
        json_data = json.load(f)
    df_EVehicleTours=json_normalize(json_data['VehicleDescriptions'])
    try:
        df_EVehicleTours=df_EVehicleTours[columns_used]
        df_EVehicleTours['time']=time_in_simulation

        path=path_dic['results']+'toure and trajectory/'
        with open(path+'df_EVehicleTours_day.csv', 'a') as f:
            df_EVehicleTours.to_csv(f, header=False, index=False,sep=';')
        return 1
    except KeyError:
        return 0


def stage_B__write_sparse_matrix(h, number_of_zones,
                     number_slow_charger_zones, number_fast_charger_zones):

    number_of_zones_with_charger=number_slow_charger_zones+number_fast_charger_zones
    num_zone_zone=number_of_zones*number_of_zones
    num_zone_charger=number_of_zones*number_of_zones_with_charger
    num_main_vars=h*(num_zone_zone+2*num_zone_charger)

    make_acc_over_zone=np.zeros([number_of_zones*h,number_of_zones*h])
    make_acc_over_zone_by_zone=np.zeros([num_main_vars,num_main_vars])
    
    h_tril=np.tril(np.ones(([h,h])))
    
    for i in range(number_of_zones):
        make_acc_over_zone[i*h:(i+1)*h,i*h:(i+1)*h]=h_tril
    for i in range(num_zone_zone+2*num_zone_charger):
        make_acc_over_zone_by_zone[i*h:(i+1)*h,i*h:(i+1)*h]=h_tril
    
    reorder_last_by_h_zones_1=np.zeros([number_of_zones,number_of_zones*h])
    reorder_last_by_h_zones_by_zones_1=np.zeros([int(num_main_vars/h),num_main_vars])
    reorder_last_by_h_zones=np.zeros([number_of_zones*h,number_of_zones*h])
    reorder_last_by_h_zones_by_zones=np.zeros([num_main_vars,num_main_vars])
        
    for i in range(number_of_zones):
        reorder_last_by_h_zones_1[i,i*h]=1
    for i in range(int(num_main_vars/h)):
        reorder_last_by_h_zones_by_zones_1[i,i*h]=1   

    reorder_last_by_h_zones[:number_of_zones,:]=reorder_last_by_h_zones_1
    reorder_last_by_h_zones_by_zones[:(num_zone_zone+2*num_zone_charger),:]=reorder_last_by_h_zones_by_zones_1
    for i in range(1,h):
        reorder_last_by_h_zones[i*number_of_zones:(i+1)*number_of_zones,:]=np.concatenate([np.zeros([number_of_zones,i]),reorder_last_by_h_zones_1],1)[:,:-i]
    for i in range(1,h):
        reorder_last_by_h_zones_by_zones[i*(num_zone_zone+2*num_zone_charger):(i+1)*(num_zone_zone+2*num_zone_charger),:]=np.concatenate([np.zeros([int(num_main_vars/h),i]),reorder_last_by_h_zones_by_zones_1],1)[:,:-i]

    out_from_zone=np.zeros([number_of_zones*h,num_main_vars])
    in_to_zone=np.zeros([number_of_zones*h,num_main_vars])
    out_from_charger=np.zeros([number_of_zones_with_charger*h,num_main_vars])
    in_to_charger=np.zeros([number_of_zones_with_charger*h,num_main_vars])
    in_to_charger_for_X=np.zeros([number_of_zones_with_charger*h,num_main_vars])
    in_to_charger_delayed=np.zeros([number_of_zones_with_charger*h,num_main_vars])
    in_to_charger=np.zeros([number_of_zones_with_charger*h,num_main_vars])
    
    out_from_zone_1=np.zeros([number_of_zones,int(num_main_vars/h)])
    in_to_zone_1=np.zeros([number_of_zones,int(num_main_vars/h)])
    out_from_charger_1=np.zeros([number_of_zones_with_charger,int(num_main_vars/h)])
    in_to_charger_1=np.zeros([number_of_zones_with_charger,int(num_main_vars/h)])

    
    zone_to_charger_t0=np.zeros([number_of_zones,num_main_vars])
    zone_to_zone_t0=np.zeros([number_of_zones,num_main_vars])
    
    for i in range(number_of_zones):
        a=num_zone_zone
        indexs=list(range(a+i*number_of_zones_with_charger,a+(i+1)*number_of_zones_with_charger))
        zone_to_charger_t0[i,indexs]=1
        
        a=0
        indexs=list(range(i*number_of_zones,(i+1)*number_of_zones))
        zone_to_zone_t0[i,indexs]=1    
    
    for i in range(number_of_zones):
        a=num_zone_zone
        indexs=list(range(i*number_of_zones,(i+1)*number_of_zones))+list(range(a+i*number_of_zones_with_charger,a+(i+1)*number_of_zones_with_charger))
        out_from_zone_1[i,indexs]=1

        a=num_zone_charger+num_zone_zone
        indexs=[number_of_zones*j+i for j in list(range(number_of_zones))]+list(range(a+i*number_of_zones_with_charger,a+(i+1)*number_of_zones_with_charger))
        in_to_zone_1[i,indexs]=1  

        
    for i in range(h):
        out_from_zone[i*number_of_zones:(i+1)*number_of_zones,i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]=out_from_zone_1
    for i in range(h-1):
        in_to_zone[(i+1)*(number_of_zones):(i+2)*(number_of_zones),i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]=in_to_zone_1
       

    for i in range(number_of_zones_with_charger):
        a=num_zone_charger+num_zone_zone
        indexs=[a+number_of_zones_with_charger*j+i for j in list(range(number_of_zones))]
        out_from_charger_1[i,indexs]=1

        a=num_zone_zone
        indexs=[a+number_of_zones_with_charger*j+i for j in list(range(number_of_zones))]
        in_to_charger_1[i,indexs]=1  

        
    for i in range(h):
        out_from_charger[i*number_of_zones_with_charger:(i+1)*number_of_zones_with_charger,i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]=out_from_charger_1
    for i in range(h-1):
        in_to_charger[(i+1)*(number_of_zones_with_charger):(i+2)*(number_of_zones_with_charger),i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]=in_to_charger_1
    for i in range(h):
        in_to_charger_for_X[(i)*(number_of_zones_with_charger):(i+1)*(number_of_zones_with_charger),i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]=in_to_charger_1

    in_to_charger=in_to_charger_for_X

    delay=min_charge_duration['slow']
    for i in range(h-delay):
        in_to_charger_delayed[(i+delay)*(number_of_zones_with_charger):(i+delay)*(number_of_zones_with_charger)+number_slow_charger_zones,
                                i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]+=in_to_charger_1[:number_slow_charger_zones,:]

    delay=min_charge_duration['fast']        
    for i in range(h-delay):
        in_to_charger_delayed[(i+delay)*(number_of_zones_with_charger)+number_slow_charger_zones:(i+delay+1)*(number_of_zones_with_charger),
                                i*int(num_main_vars/h):(i+1)*int(num_main_vars/h)]+=in_to_charger_1[number_slow_charger_zones:,:]

        
    sum_over_chargers={}    
    sum_over_chargers['slow']=np.zeros([h,h*number_of_zones_with_charger])
    sum_over_chargers['fast']=np.zeros([h,h*number_of_zones_with_charger])
    
    
    for i in range(h):
        sum_over_chargers['slow'][i,range(i*number_of_zones_with_charger,i*number_of_zones_with_charger+number_slow_charger_zones)]=1
    for i in range(h):
        sum_over_chargers['fast'][i,range(i*number_of_zones_with_charger+number_slow_charger_zones,(i+1)*number_of_zones_with_charger)]=1
        

    
    sparse_make_acc_over_zone_by_zone=csr_matrix(make_acc_over_zone_by_zone)
    sparse_reorder_last_by_h_zones_by_zones=csr_matrix(reorder_last_by_h_zones_by_zones)
    sparse_out_from_zone=csr_matrix(out_from_zone)
    sparse_in_to_zone=csr_matrix(in_to_zone)
    sparse_out_from_charger=csr_matrix(out_from_charger)
    sparse_in_to_charger=csr_matrix(in_to_charger)
    sparse_in_to_charger_for_X=csr_matrix(in_to_charger_for_X)
    sparse_sum_over_chargers={}
    sparse_sum_over_chargers_slow=csr_matrix(sum_over_chargers['slow'])
    sparse_sum_over_chargers_fast=csr_matrix(sum_over_chargers['fast'])
    sparse_zone_to_charger_t0=csr_matrix(zone_to_charger_t0)
    sparse_zone_to_zone_t0=csr_matrix(zone_to_zone_t0)
    sparse_in_to_charger_delayed=csr_matrix(in_to_charger_delayed)
    sparse_make_acc_over_zone=csr_matrix(make_acc_over_zone)
    sparse_reorder_last_by_h_zones=csr_matrix(reorder_last_by_h_zones)
    sparse_in_to_charger_delayed=csr_matrix(in_to_charger_delayed)
    path=path_dic['scenario']
    for mat in ['sparse_in_to_charger_delayed',
                'sparse_reorder_last_by_h_zones',
                'sparse_make_acc_over_zone',
                'sparse_make_acc_over_zone_by_zone',
               'sparse_reorder_last_by_h_zones_by_zones',
               'sparse_out_from_zone',
               'sparse_in_to_zone',
               'sparse_out_from_charger',
               'sparse_in_to_charger',
                'sparse_in_to_charger_for_X',
               'sparse_zone_to_charger_t0',
               'sparse_zone_to_zone_t0',
               'sparse_in_to_charger_delayed',
               'sparse_sum_over_chargers_slow',
               'sparse_sum_over_chargers_fast']:
        save_npz(path+mat+".npz", eval(mat))




def stage_B__get_A_b(h, number_of_zones, number_chargers_in_zones_with_charger,
                     number_slow_charger_zones, number_fast_charger_zones, 
                     initial_v0,pick_ups_h,drop_offs_h , 
                     X_h,Y_h,
                     cost,must_cut_charging_h, can_cut_charging_h, 
                     avilable_0, chargable_0, penalty):
    assert np.shape(pick_ups_h)[0]==h,'wrong shape for pick up and drop off'
    A_ub_dictionary={}
    b_ub_dictionary={}
    A_eq_dictionary={}
    b_eq_dictionary={}
    min_charge_duration={'slow':3, 'fast': 3}
    number_of_zones_with_charger=number_slow_charger_zones+number_fast_charger_zones
    drop_offs_h_devided_by_P=drop_offs_h/np.maximum(np.sum(pick_ups_h,1),1)[:,np.newaxis]

    drop_offs_h_devided_by_P_shifted=drop_offs_h_devided_by_P
    drop_offs_h_devided_by_P_shifted=np.reshape(drop_offs_h_devided_by_P_shifted,[h*number_of_zones,1],'F')
    drop_offs_h_shifted=np.concatenate([np.zeros([1,number_of_zones]), drop_offs_h])[:-1,:]
    pick_ups=np.reshape(pick_ups_h,[h*number_of_zones],'F')
    drop_offs_shifted=np.reshape(drop_offs_h_shifted,[h*number_of_zones],'F')
    
    v0_zones=initial_v0[:number_of_zones]
    v0_chargers=initial_v0[number_of_zones:]
    v0_zones_h=np.tile(v0_zones,h)
    v0_chargers_h=np.tile(v0_chargers,h)
    num_zone_zone=number_of_zones*number_of_zones
    num_zone_charger=number_of_zones*number_of_zones_with_charger
    num_main_vars=h*(num_zone_zone+2*num_zone_charger)
    
    start=time.time()
    sparse_matrixs={}
    path=path_dic['scenario']
    for mat in ['sparse_in_to_charger_delayed',
                'sparse_reorder_last_by_h_zones',
                'sparse_make_acc_over_zone',
                'sparse_make_acc_over_zone_by_zone',
               'sparse_reorder_last_by_h_zones_by_zones',
               'sparse_out_from_zone',
               'sparse_in_to_zone',
               'sparse_out_from_charger',
               'sparse_in_to_charger',
                'sparse_in_to_charger_for_X',
               'sparse_zone_to_charger_t0',
               'sparse_zone_to_zone_t0',
               'sparse_in_to_charger_delayed',
               'sparse_sum_over_chargers_slow',
               'sparse_sum_over_chargers_fast']:
        sparse_matrixs[mat]= load_npz(path+mat+".npz") 
    #constraint 1    #num vehicles in zones posetive #@h
    sparse_matrixs['sparse_sum_over_chargers']={'slow':sparse_matrixs['sparse_sum_over_chargers_slow'],'fast':sparse_matrixs['sparse_sum_over_chargers_fast']}

    sparse_A_ub=csr_matrix(sparse_matrixs['sparse_out_from_zone']-sparse_matrixs['sparse_in_to_zone']).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    b_ub=sparse_matrixs['sparse_reorder_last_by_h_zones'].dot(sparse_matrixs['sparse_make_acc_over_zone']).dot(drop_offs_shifted-pick_ups)+v0_zones_h
    A_ub_dictionary['num_vehicles_in_zones']=sparse_A_ub
    b_ub_dictionary['num_vehicles_in_zones']=b_ub
    #del 
    
    #constraint 2    #num vehicles in chargers posetive #@h
    sparse_A_ub=csr_matrix(sparse_matrixs['sparse_out_from_charger'].todense()-sparse_matrixs['sparse_in_to_charger'].todense()).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    b_ub=v0_chargers_h
    A_ub_dictionary['num_vehicles_in_charger_zones']=sparse_A_ub
    b_ub_dictionary['num_vehicles_in_charger_zones']=b_ub
 
    
    #constraint 3    #chargable vehicles #@t0
    sparse_A_ub=sparse_matrixs['sparse_zone_to_charger_t0'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    b_ub=chargable_0
    A_ub_dictionary['chargable_0']=sparse_A_ub
    b_ub_dictionary['chargable_0']=b_ub

    
    #constraint 4    #avilable vehicles #@t0
    sparse_A_ub=sparse_matrixs['sparse_zone_to_zone_t0'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    b_ub=np.maximum(avilable_0-sparse_matrixs['sparse_reorder_last_by_h_zones'].dot(pick_ups)[:number_of_zones],np.zeros([number_of_zones]))
    A_ub_dictionary['avilable_0']=sparse_A_ub
    b_ub_dictionary['avilable_0']=b_ub  
    
    #constraint 5    #must cut of charging
    sparse_A_ub=(-1*sparse_matrixs['sparse_out_from_charger']).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    b_ub=-1*np.reshape(must_cut_charging_h,[h*number_of_zones_with_charger],'F')
    A_ub_dictionary['must_cut_charging_h']=sparse_A_ub
    b_ub_dictionary['must_cut_charging_h']=b_ub

    #constraint 5    #can cut of charging
    sparse_A_ub=csr_matrix(sparse_matrixs['sparse_out_from_charger'].todense()-sparse_matrixs['sparse_in_to_charger_delayed'].todense()).dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    b_ub=np.reshape(can_cut_charging_h,[h*number_of_zones_with_charger],'F')
    A_ub_dictionary['can_cut_charging_h']=sparse_A_ub
    b_ub_dictionary['can_cut_charging_h']=b_ub

    #constraint 6    #in charge vehicles
    sparse_A_eq_base=csr_matrix(-1*sparse_matrixs['sparse_out_from_charger'].todense()+
                                                sparse_matrixs['sparse_in_to_charger'].todense()).dot(
                                                sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones']).dot(
                                                sparse_matrixs['sparse_make_acc_over_zone_by_zone'])
    for type_ in ['slow','fast']:
        sparse_A_eq=sparse_matrixs['sparse_sum_over_chargers'][type_].dot(sparse_A_eq_base)
        b_eq=Y_h[type_]-sparse_matrixs['sparse_sum_over_chargers'][type_].dot(v0_chargers_h)
        A_eq_dictionary['in_charge_'+type_]=sparse_A_eq
        b_eq_dictionary['in_charge_'+type_]=b_eq
    
    
    
    
    #constraint 7    #going to charge vehicles 
    sparse_A_ub_base=-1*sparse_matrixs['sparse_in_to_charger_for_X'].dot(sparse_matrixs['sparse_reorder_last_by_h_zones_by_zones'])
    
    for type_ in ['slow','fast']:
        sparse_A_ub=sparse_matrixs['sparse_sum_over_chargers'][type_].dot(sparse_A_ub_base)
        b_ub=-1*X_h[type_]
        A_ub_dictionary['go_to_charge_'+type_]=sparse_A_ub
        b_ub_dictionary['go_to_charge_'+type_]=b_ub
    #del 

    #constraint 8    #on number of chargers
    sparse_A_ub=A_ub_dictionary['num_vehicles_in_charger_zones']*-1
    b_ub=-1*v0_chargers_h+np.tile(np.array(number_chargers_in_zones_with_charger),h)
    A_ub_dictionary['charger_capacity']=sparse_A_ub
    b_ub_dictionary['charger_capacity']=b_ub
    #del 
    
    
    #add slak variables for flow and in charge
    

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
    
 

    #add zeros to rightside of the rest of constraints, if not the right size
    for key in list(A_ub_dictionary.keys()):
        A_ub=A_ub_dictionary[key].todense()
        if np.shape(A_ub)[1]<num_main_vars+h*number_of_zones+6*h+number_of_zones_with_charger*h:
            sparse_zero=csr_matrix(np.zeros([np.shape(A_ub)[0],h*number_of_zones+6*h+number_of_zones_with_charger*h]))
            A_ub_dictionary[key]=hstack([A_ub_dictionary[key], sparse_zero])
           
##    print([[a,A_ub_dictionary[a].shape] for a in list(A_ub_dictionary.keys())])
    A_ub=vstack(list(A_ub_dictionary.values()))
    b_ub=np.concatenate(list(b_ub_dictionary.values()))
                   
    A_eq=vstack(list(A_eq_dictionary.values()))
    b_eq=np.concatenate(list(b_eq_dictionary.values()))                  
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
    S_ub=coo_matrix(A_ub).tocsr()
    S_eq=coo_matrix(A_eq).tocsr()

    return S_ub, b_ub,S_eq, b_eq, c, A_ub_dictionary,A_eq_dictionary, b_ub_dictionary, b_eq_dictionary


def stage_B__solve_gurobi(S_ub, b_ub, S_eq, b_eq, c, B_time_limit):

    
    start = time.time()
    model = gurobipy.Model()
    model.setParam("OutputFlag",0)

    rows_ub,rows_eq, cols = len(b_ub),len(b_eq), len(c)
    x = []
    x_answer=np.zeros([cols])
    
    for j in range(cols):
      x.append(model.addVar(lb=0, obj=c[j], vtype=gurobipy.GRB.INTEGER))
    model.update()
    start__ = time.time()

    # iterate over the rows of S adding each row into the model
    for i in range(rows_ub):
      start = S_ub.indptr[i]
      end   = S_ub.indptr[i+1]
      variables = [x[j] for j in S_ub.indices[start:end]]
      coeff     = S_ub.data[start:end]
      expr = gurobipy.LinExpr(coeff, variables)
      model.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=b_ub[i])
    model.update()      
    
    for i in range(rows_eq):
      start = S_eq.indptr[i]
      end   = S_eq.indptr[i+1]
      variables = [x[j] for j in S_eq.indices[start:end]]
      coeff     = S_eq.data[start:end]
      expr = gurobipy.LinExpr(coeff, variables)
      model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b_eq[i])
    model.update()  
    
    
    start = time.time()
    model.ModelSense = gurobipy.GRB.MINIMIZE

    model.setParam("MIPFocus", 1)  
    model.setParam("MIPGap", 0.10)
    model.setParam('TimeLimit',B_time_limit)
    model.update()
    model.optimize()
##    print("B MIP gap value: %f" % model.MIPGap)

    
    if (model.status == gurobipy.GRB.Status.OPTIMAL) or (model.status == gurobipy.GRB.Status.TIME_LIMIT):
        i=0
        for v in model.getVars():
            x_answer[i]=v.x
            i+=1

        start = time.time()
        return x_answer, 1
    else:
        print('status is:',model.status, 'returing model instead' )
        return model, 0

    

class XServer:    
    def make_route_request(self, profile, waypoints,calc_mode="HIGH_PERFORMANCE_ROUTING"):

        request = {"waypoints": [],
                    "routeOptions": {
                        "polylineOptions": {
                          "elevations": False #True
                        }
                      },  
                   "resultFields": {
                       "polyline": False,
                      "report": True,
                       "segments": {
                           "enabled": False,
#                            "descriptors": True,
#                          "polyline": False,
#                            "roadAttributes": True
                                   },
                       "waypoints": True
                                    },
                        "routeOptions": {
                                "routingType": calc_mode,
                            
                        "geographicRestrictions": {
                              "allowedCountries": [
                                "ES"
                              ]
                            },

                                },
                   "storedProfile": "{}".format(profile),   
                     "scope": "barcelona",
                   "userLogs": [],
                   "coordinateFormat": "EPSG:76131"
                   }
        
        request["waypoints"]  =[
            {"$type": "OnRoadWaypoint", "location": {"coordinate": {"y": stop[1], "x": stop[0]}}} for stop in list(waypoints)]
        
        
        return request   

    def send_route_request(self, request):
        
        
        json_data = json.dumps(request)

        UrlCalculateRoute = "http://localhost:50000" + "/services/rs/XRoute/experimental/calculateRoute"
        header = {"content-type": "application/json;charset=utf-8"}
        json_resp = post(url=UrlCalculateRoute, data=json_data, headers=header)

        if json_resp.status_code == 200:
            response = json_resp.text
            pyres = json.loads(response)

        else:
            if LOG==1:
                print('xs2 Request failed 1')
                print(json.loads(json_resp.text))
            return -99,-99,-99
        distance = 0
        if "$type" in pyres:

            # read out the data of xs-response

            distance = pyres.get("distance")
            travel_time = pyres.get("travelTime")
            report = pyres.get("report")
            way_points = pyres.get("waypoints")

        else:
            "Couldn't read data from xs"
               


        return distance, travel_time, way_points#, report

    
def link_consumption (distance, driving_range=DrivingRange):
    consumption=100*(distance/(driving_range*1000))
    return consumption



def read_simulation_files():
    def get_futur_locations():

        def tours2waypoints(Stops):
            dic_Stops=Stops
            waypoints=[[stop['Location']['X']/10,stop['Location']['Y']/10] for stop in dic_Stops if stop['EstimatedArrivalTime']['Seconds']>time_in_simulation]
            return waypoints
        def get_active_tours(NextStopIndex,x):

            dic_tours=x
            dic_tours_active=[k for k in dic_tours if k['DropoffIndex']>=NextStopIndex]
            return dic_tours_active
        def get_trip_ids(x):
            list_TripId=[k['TripId'] for k in x]
            return list_TripId
        path=path_dic['from_simulator']
        with open(path+'EVehicleTours.json') as f:
            json_data = json.load(f)

        df_EVehicleTours=json_normalize(json_data['VehicleDescriptions'])

        columns_used=[
         'CurrentLocation.X',
         'CurrentLocation.Y',
         'Tour.NextStopIndex',
         'Tour.Stops',
         'Tour.Trips',
         'VehicleId']  

        df_future_locations=df_EVehicleTours[columns_used]

        df_future_locations.loc[:,'Trips_dic']=df_future_locations[['Tour.NextStopIndex','Tour.Trips']].apply(lambda row:get_active_tours(row['Tour.NextStopIndex'],row['Tour.Trips']),axis=1)  

        df_future_locations.loc[:,'TripIds']=df_future_locations['Trips_dic'].apply(lambda x:get_trip_ids(x))  

        df_future_locations.loc[:,'waypoints']=df_future_locations[[
         'CurrentLocation.X',
         'CurrentLocation.Y',
         'Tour.Stops']].apply(lambda row: [[row['CurrentLocation.X']/10,row['CurrentLocation.Y']/10]]+tours2waypoints(row['Tour.Stops']),axis=1)
        return df_future_locations



    def get_past_locations():
        def add_vehid( df,vehid):
            df.insert(0, 'VehicleId', vehid, allow_duplicates=False)
            return df
        headers = ['Datetime', 'Lon', 'Lat']
        dtypes = {'Datetime': 'str', 'Lon': 'str', 'Lat': 'str'}
        parse_dates = ['Datetime']
        path=path_dic['from_simulator']
        if time_in_simulation==time_step_online:
            
            df_EVehicleTrejectory = pd.concat([add_vehid(pd.read_csv(f,sep=',',header=None,skiprows=1, names=headers, dtype=dtypes, parse_dates=parse_dates) , int(os.path.basename(f)[7:-4])) for f in glob.glob(path+'vehicle*.csv')], ignore_index = True)   
        else:
            df_EVehicleTrejectory = pd.concat([add_vehid(pd.read_csv(f,sep=',',header=None,skiprows=int(fleet_attribute.loc[fleet_attribute['VehicleId']==int(os.path.basename(f)[7:-4])]['last_row_trejectory'].item()),
                                                                     names=headers, dtype=dtypes, parse_dates=parse_dates) , int(os.path.basename(f)[7:-4])) for f in glob.glob(path+'vehicle*.csv')], ignore_index = True)   
        if len(df_EVehicleTrejectory)==0:
            os._exit(0)
        df_EVehicleTrejectory['waypoints']=df_EVehicleTrejectory.apply(lambda row: [row['Lon'],row['Lat']], axis=1)
        df_past_locations=pd.DataFrame(data={'VehicleId':df_future_locations['VehicleId']})
        df_past_locations=pd.merge(df_past_locations,df_EVehicleTrejectory,on='VehicleId',how='left')
        df_past_locations=df_past_locations.groupby(['VehicleId'])['waypoints'].apply(list).reset_index()
        return df_past_locations


    x_server = XServer()
    def get_cunsumption_TT(waypoints, x_server, link_consumption,calc_mode="HIGH_PERFORMANCE_ROUTING"):
        if len(waypoints)<2:
            return [0, 0]
        profile="car"
        route_request=x_server.make_route_request( profile, waypoints,calc_mode)
        distance, travel_time, _=x_server.send_route_request(route_request)
        cunsumption=link_consumption(distance)
        return [cunsumption, travel_time]
    df_future_locations=get_futur_locations()
    df_past_locations= get_past_locations()
    x_server = XServer()

    df_past_locations.loc[:,'consumption_TT']=df_past_locations.apply(lambda row:
                                            get_cunsumption_TT(row['waypoints'],x_server,link_consumption,"CONVENTIONAL"), axis=1)
    df_future_locations.loc[:,'consumption_TT']=df_future_locations.apply(lambda row:
                                            get_cunsumption_TT(row['waypoints'],x_server,link_consumption), axis=1)
    for df in [df_past_locations,df_future_locations]:
        df.loc[:,'consumption']=df.loc[:,'consumption_TT'].apply(lambda x:x[0])
        df.loc[:,'TT']=df.loc[:,'consumption_TT'].apply(lambda x:x[1])
    df_past_locations['number_of_rows']=df_past_locations['waypoints'].apply(lambda x:len(x))    
    df_past=df_past_locations[['VehicleId','consumption','number_of_rows']]
    df_past.rename(columns={'consumption': 'consumption_past'}, inplace=True)   
    df_future_locations.loc[:,'location']=df_future_locations.loc[:,['CurrentLocation.X','CurrentLocation.Y']].apply(lambda row:[row['CurrentLocation.X'],row['CurrentLocation.Y']],axis=1)
    df_future_locations.loc[:,'location_']=df_future_locations.loc[:,'waypoints'].apply(lambda x:x[-1])
    df_future=df_future_locations[['VehicleId','consumption','TT','location','location_','TripIds','waypoints']]
    df_future.rename(columns={'consumption': 'consumption_future','TT': 'EAT'}, inplace=True) 
    

    return df_past, df_future


if not lazy_charging:
    def make_fleet_attribute(VehicleId_list):
        zeros=np.zeros([feet_size])
        ones=np.ones([feet_size])
        fleet_attribute=pd.DataFrame(data={'VehicleId':VehicleId_list,
                                           'SoC_now':100*ones, 
                                           'charging_planed_slow':zeros,
                                           'charging_planed_fast':zeros,
                                           'relocation_planed':zeros,
                                             'Expected_start_charging_time':zeros,
                                           'Expected_arrival_on_relocation':zeros,
                                           'in_charge_slow':zeros, 
                                           'in_charge_fast':zeros, 
                                           'charger':zeros, 
                                           'charge_duration':zeros,
                                           'trip_id':zeros,
                                           'last_row_trejectory':zeros})
        return fleet_attribute



    def make_files_to_track_performance(files_to_track_performance,task,to_append={}):
        
        path=path_dic['results']
        for name in list(files_to_track_performance.keys()): 
            if task=='make':
                files_to_track_performance[name]['df']=pd.DataFrame(columns=files_to_track_performance[name]['columns'])
                files_to_track_performance[name]['df'].to_csv(path+name+'.csv',sep=';',index=False)
            if task=='read':
                files_to_track_performance[name]['df']=pd.read_csv(path+name+'.csv',sep=';')
            if task=='write':
                with open(path+name+'.csv', 'a') as f:
                    if len(files_to_track_performance[name]['df'])>0:
                        files_to_track_performance[name]['df'].to_csv(f, header=False, index=False,sep=';')
        if task=='update':
            for name in list(to_append.keys()): 
                to_append[name]['time']=time_in_simulation
                to_append[name]=to_append[name][files_to_track_performance[name]['columns']]
                files_to_track_performance[name]['df']=to_append[name]
        return files_to_track_performance

    def read_static_data():
        path=path_dic['scenario']
        if STAGE==3:
            charger_attribute=pd.read_csv(path+'charger_attribute.csv',sep=';')
            charger_attribute['charger_location']=charger_attribute['charger_location'].apply(lambda x:literal_eval(x))
        else:
            charger_attribute=pd.DataFrame(columns= ['charger_id','capacity','fast','slow','charger_location','charger_zone']).astype(int)
        zone_attribute=pd.read_csv(path+'zone_attribute.csv',sep=';')
        zone_attribute['zone_center_coord']=zone_attribute['zone_center_coord'].apply(lambda x:literal_eval(x))
        if STAGE==3:
            cost_charger_to_zone_center=pd.read_csv(path+'cost_charger_to_zone_center.csv',sep=';')
        else:
            cost_charger_to_zone_center=pd.DataFrame(columns= ['charger','zone','distance','travel_time']).astype(int)
        df=pd.read_csv(path+"Zones_ptv_mercator.csv",sep=';')
        df['PTV_mercator_list']=df['PTV_mercator_list'].apply(lambda x:literal_eval(x))
        geometry = [Polygon(x) for x in df.PTV_mercator_list.tolist()]
        crs = {'init': 'ptv_mercator'}
        zones = GeoDataFrame(df, crs=crs, geometry=geometry)
        list_of_chargers={'slow':charger_attribute.loc[charger_attribute['slow']==1,'charger_id'].tolist(),
                          'fast':charger_attribute.loc[charger_attribute['fast']==1,'charger_id'].tolist()}
        if STAGE==3:
            path=path_dic['scenario']
            with open(path+'DailyPlan.json', "r") as f:
                DailyPlan=json.load(f)
            zero_list=list(np.zeros(H))
            DailyPlan['X_day']['slow']+=zero_list
            DailyPlan['X_day']['fast']+=zero_list
            DailyPlan['Y_day']['slow']+=zero_list
            DailyPlan['Y_day']['fast']+=zero_list
            DailyPlan['mean_charge_cut_charge']+=zero_list 
            DailyPlan['mean_charge_go_to_charge']+=zero_list
            if SoC_in_assignment:
##                path=path_dic['scenario']
##                with open(path+'soc_metric.json', "r") as f:
##                    soc_metric=json.load(f)
##                soc_metric_t={}
##                for soc in list(soc_metric.keys()):
##                    soc_metric_t[soc]=soc_metric[soc][time_step]
                path=path_dic['scenario']
                with open(path+'metric_input_data.pickle', 'rb') as handle:
                    metric_input_data = pickle.load(handle)
                l = locals()
                for var in list(metric_input_data.keys()):
                    exec(var + '=metric_input_data[var]') 
                convert_nan=lambda x: np.nan_to_num(x.astype('float'))

                for_above=0
                free_capacity, capacity_of_above, delta_mat, number_of_vehicles_above=metric_soc_CG_t(time_step,locals()['number_of_time_slots'],locals()['stop_charge'] ,locals()['go_charge'], 
                                                                                                    locals()['mean_charge_cut_charge'],locals()['mean_charge_in_charge'] , 
                                                                                                    fleet_attribute['SoC_now'].values, fleet_attribute['in_charge_slow'].values+fleet_attribute['in_charge_fast'].values,
                                                                                                      locals()['decharge_rate'], 
                                                                                                    locals()['demand_sum_over_zones'], 1, for_above,
                                                                                                    locals()['charge_go_to_charge_all'], locals()['charge_cut_charge_all'])

                free_capacity=convert_nan(free_capacity)
                capacity_of_above=convert_nan(capacity_of_above)
                free_capacity=np.minimum(free_capacity,capacity_of_above)
                metric=(capacity_of_above-free_capacity)/(delta_mat*number_of_vehicles_above)  
                metric_0=np.nanmax(metric) 
                number_of_vehicles_above_0=number_of_vehicles_above
                metric_dic={}
                for for_above in [80,60,40,20,0]:
                    
                    free_capacity, capacity_of_above, delta_mat, number_of_vehicles_above=metric_soc_CG_t(time_step,locals()['number_of_time_slots'],locals()['stop_charge'] ,locals()['go_charge'], 
                                                                                                    locals()['mean_charge_cut_charge'],locals()['mean_charge_in_charge'] , 
                                                                                                    fleet_attribute['SoC_now'].values, fleet_attribute['in_charge_slow'].values+fleet_attribute['in_charge_fast'].values,
                                                                                                      locals()['decharge_rate'], 
                                                                                                    locals()['demand_sum_over_zones'], 1, for_above,
                                                                                                    locals()['charge_go_to_charge_all'], locals()['charge_cut_charge_all'])
                    free_capacity=convert_nan(free_capacity)
                    capacity_of_above=convert_nan(capacity_of_above)
                    free_capacity=np.minimum(free_capacity,capacity_of_above)
                    metric=((capacity_of_above-free_capacity)/(delta_mat*number_of_vehicles_above))
                    normal_metric=(np.nanmax(metric)/metric_0)/(number_of_vehicles_above/number_of_vehicles_above_0).flatten()
                    normal_metric[np.isnan(normal_metric)]=1
                    normal_metric=np.maximum(normal_metric,1)
                    metric_dic[for_above]=min(normal_metric[0],5)
                print(metric_dic)
                path=path_dic['auxilary']
                with open(path+'soc_metric_t.json', "w") as f:
                    json.dump(metric_dic,f)   
        else:
            zero_list=list(np.zeros(number_of_steps_in_day+H))
            X_day={'slow':zero_list,'fast':zero_list}
            Y_day={'slow':zero_list,'fast':zero_list}
            DailyPlan={'X_day':X_day,
               'Y_day':Y_day,
               'mean_charge_cut_charge':zero_list,
               'mean_charge_go_to_charge':zero_list}  
        #zone counts
        charger_zone_attribute=charger_attribute.groupby(['charger_zone'])['capacity'].sum().reset_index().sort_values(by='charger_zone')
        charger_zone_attribute.rename(columns={'charger_zone':'zone_'},inplace=True)
        number_of_zones=len(zone_attribute)
        number_chargers_in_slow_charger_zones=charger_zone_attribute[charger_zone_attribute['zone_']<10000].as_matrix(
                                                                                        columns=['capacity'])[:,0]
        number_chargers_in_fast_charger_zones=charger_zone_attribute[charger_zone_attribute['zone_']>=10000].as_matrix(
                                                                                        columns=['capacity'])[:,0]
        number_chargers_in_zones_with_charger=np.concatenate([number_chargers_in_slow_charger_zones,
                                                              number_chargers_in_fast_charger_zones],0)
        number_slow_charger_zones=len(charger_zone_attribute[charger_zone_attribute['zone_']<10000])
        number_fast_charger_zones=len(charger_zone_attribute[charger_zone_attribute['zone_']>=10000])

        number_of_zones_with_charger=number_slow_charger_zones+number_fast_charger_zones 
        number_of_all_zones=number_of_zones+number_slow_charger_zones+number_fast_charger_zones
        return (charger_attribute,charger_zone_attribute,number_chargers_in_zones_with_charger, 
                number_slow_charger_zones,number_fast_charger_zones,number_of_zones_with_charger,
                zone_attribute,number_of_zones,number_of_all_zones,
                cost_charger_to_zone_center, zones, DailyPlan, list_of_chargers)




    def update_fleet_attribute_with_simulation_data(fleet_attribute,files_to_track_performance): 
        if time_step<len(DailyPlan['mean_charge_cut_charge']):
            
            UP_SoC_for_charging=min(DailyPlan['mean_charge_go_to_charge'][time_step]+15,np.nanmax(DailyPlan['mean_charge_go_to_charge']))
            
            LB_SoC_for_stop_charging=DailyPlan['mean_charge_cut_charge'][time_step]-25
            UP_SoC_for_stop_charging=min(DailyPlan['mean_charge_cut_charge'][time_step]+25,np.nanmax(DailyPlan['mean_charge_cut_charge']))
        else:
            UP_SoC_for_charging=50
            LB_SoC_for_stop_charging=0
            UP_SoC_for_stop_charging=90
        if np.isnan(UP_SoC_for_charging)==1:
            UP_SoC_for_charging=70
        if np.isnan(LB_SoC_for_stop_charging)==1:
            LB_SoC_for_stop_charging=min(40,UP_SoC_for_charging+30)
            UP_SoC_for_stop_charging=95

        fleet_attribute=pd.merge(fleet_attribute, df_past,how='left',on='VehicleId')
        fleet_attribute=pd.merge(fleet_attribute, df_future,how='left',on='VehicleId')

        fleet_attribute['last_row_trejectory']+=fleet_attribute['number_of_rows']
        fleet_attribute['SoC_now']=fleet_attribute['SoC_now']-fleet_attribute['consumption_past']
        
        fleet_attribute.drop(['consumption_past','consumption_future'],axis=1)
        ## increased by charging
        fleet_attribute['time_in_charge']=time_step_online*(fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast'])
        fleet_attribute['time_in_charge']+=fleet_attribute.apply(lambda row: np.maximum(time_in_simulation-
                                                                                        row['Expected_start_charging_time'],0)*
                                                                 (row['charging_planed_slow']+row['charging_planed_fast']),axis=1)
        fleet_attribute['u_slow']=fleet_attribute.apply(lambda  row: np.maximum(np.minimum(row['SoC_now']+row['in_charge_slow']*
                                                                                row['time_in_charge']*charging_rate[0]-80,
                                                                       row['in_charge_slow']*row['time_in_charge']*charging_rate[0]),0) ,axis=1)
        fleet_attribute['charge_gained']=fleet_attribute.apply(lambda  row: (row['time_in_charge']*charging_rate[0]-row['u_slow']+
                                                                             (charging_rate[1]/charging_rate[0])*row['u_slow'])
                                                               *row['in_charge_slow'],axis=1)
        
        charge_gained_slow=fleet_attribute['charge_gained'].sum()
        fleet_attribute['u_fast']=fleet_attribute.apply(lambda  row: np.maximum(np.minimum(row['SoC_now']+row['in_charge_fast']*
                                                                                row['time_in_charge']*charging_rate[2]-80,
                                                                       row['in_charge_fast']*row['time_in_charge']*
                                                                                charging_rate[2]),0),axis=1)
        fleet_attribute['charge_gained']+=fleet_attribute.apply(lambda  row: (row['time_in_charge']*charging_rate[2]-row['u_fast']+
                                                                              (charging_rate[3]/charging_rate[2])*row['u_fast'])
                                                                *row['in_charge_fast'],axis=1)
        charge_gained_fast=fleet_attribute['charge_gained'].sum()-charge_gained_slow
        performance_update={}
        performance_update['charge_gained']=pd.DataFrame(data={'slow':[charge_gained_slow],'fast':[charge_gained_fast]})
        files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)

        # Update state of in charge
        fleet_attribute['in_charge_slow']+=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_slow'],axis=1)
        fleet_attribute['charging_planed_slow']-=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_slow'],axis=1)\
        

        fleet_attribute['in_charge_fast']+=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_fast'],axis=1)
        fleet_attribute['charging_planed_fast']-=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_fast'],axis=1)

        fleet_attribute['relocation_planed']-=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_arrival_on_relocation']>0)*
                                                                        row['relocation_planed'],axis=1)
        
        fleet_attribute.loc[(fleet_attribute['charging_planed_slow']==0)&(fleet_attribute['charging_planed_fast']==0),'Expected_start_charging_time']=0
        fleet_attribute.loc[(fleet_attribute['charging_planed_slow']==0)&(fleet_attribute['charging_planed_fast']==0)&
                            (fleet_attribute['in_charge_slow']==0)&(fleet_attribute['in_charge_fast']==0),'charger']=0
        fleet_attribute.loc[(fleet_attribute['relocation_planed']==0),'Expected_arrival_on_relocation']=0
        fleet_attribute.loc[((fleet_attribute['charging_planed_slow']==1)+
                            (fleet_attribute['charging_planed_fast']==1)+
                            (fleet_attribute['in_charge_slow']==1)+
                            (fleet_attribute['in_charge_fast']==1)+
                            (fleet_attribute['relocation_planed']==1))==0,'trip_id']=0

        fleet_attribute['SoC_now']=fleet_attribute['SoC_now']+fleet_attribute['charge_gained']
        fleet_attribute['SoC_now']=fleet_attribute['SoC_now'].apply(lambda x:min(100,x))
        fleet_attribute['SoC_']=fleet_attribute['SoC_now']-fleet_attribute['consumption_future']

        fleet_attribute['update_arrival_at_charger']=fleet_attribute.apply(lambda row:(time_in_simulation+row['EAT']>
                                                                      row['Expected_start_charging_time'])*(row['charging_planed_slow']+
                                                                                                            row['charging_planed_fast']==1),axis=1)

        fleet_attribute.loc[fleet_attribute['update_arrival_at_charger']==1,'Expected_start_charging_time']=fleet_attribute.loc[
            fleet_attribute['update_arrival_at_charger']==1,'EAT'].values +time_in_simulation    
        #check for faults
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['in_charge_fast']).max()<=1, 'in charge not right'
        assert  (fleet_attribute['charging_planed_fast']+ fleet_attribute['charging_planed_slow']+ fleet_attribute['relocation_planed']).max()<=1, 'planned value not right'
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['charging_planed_slow']).max()<=1, 'in charge or planned value not right'
        if STAGE==3:
            if fleet_attribute['SoC_now'].max()>100:
                fleet_attribute.to_csv('fleet_attribute_check_for_errors.csv',sep=';',index=False)
                assert  fleet_attribute['SoC_now'].max()<=100, 'SoC over 100'
            if fleet_attribute['SoC_now'].min()<0:
                print('following vehicles have SoC below 0: ',fleet_attribute.loc[fleet_attribute['SoC_now']<0,'VehicleId'].tolist())    


        fleet_attribute['charge_left_to_full']=fleet_attribute.apply(lambda row: (UP_SoC_for_stop_charging-row['SoC_now'])*(row['in_charge_fast']+
                                                                                                       row['in_charge_slow']),
                                                                                                       axis=1)
        fleet_attribute['time_to_full_charge']=fleet_attribute.apply(lambda row: 
                                               (np.maximum(row['charge_left_to_full']-20,0)/ 
                                            (max(row['in_charge_fast']*charging_rate[2]+
                                               row['in_charge_slow']*charging_rate[0],min(charging_rate))) +
                                            np.minimum(row['charge_left_to_full'],20)/
                                             max(row['in_charge_fast']*charging_rate[3]+
                                              row['in_charge_slow']*charging_rate[1],min(charging_rate)))*(row['in_charge_slow']+row['in_charge_fast'])
                                                                     ,axis=1)

        fleet_attribute['time_to_full_charge']=fleet_attribute.apply(lambda row: round((row['time_to_full_charge']/time_step_B)-0.5),axis=1)                                                             
        fleet_attribute['arrival_time_in_B_steps']=fleet_attribute.apply(lambda row: round((row['EAT']/time_step_B)-0.5),axis=1)                                                                                                                                                           
        fleet_attribute['can_charge']=fleet_attribute.apply(lambda row: (row['SoC_']<UP_SoC_for_charging)
                                                            *(row['arrival_time_in_B_steps']==0)
                                                            *(row['in_charge_slow']+row['in_charge_fast']==0)
                                                           *( row['charging_planed_slow']+row['charging_planed_fast']+row['relocation_planed']==0 ),axis=1)

        if STAGE==3 and SoC_in_assignment:
            fleet_attribute['can_trip_0']=fleet_attribute.apply(lambda row: (row['SoC_']>LB_SoC_for_trip)*(row['arrival_time_in_B_steps']==0)
                                                                *(row['in_charge_slow']+row['in_charge_fast']==0)
                                                               *( row['charging_planed_slow']+row['charging_planed_fast']+row['relocation_planed']==0 ),axis=1)
        else:
            fleet_attribute['can_trip_0']=fleet_attribute.apply(lambda row: (row['arrival_time_in_B_steps']==0)*(row['relocation_planed']==0) ,axis=1)

        fleet_attribute['all_vehicles']=1
        for i in range(H):                                                             
            fleet_attribute['should_stop_charging_before_'+str(i)]=(fleet_attribute['time_to_full_charge']<i)*(fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast'])

        fleet_attribute.loc[fleet_attribute['SoC_now']>=UP_SoC_for_stop_charging,'should_stop_charging_before_'+str(0)]=1
        def get_charge_in_t_min(SoC_now,time_in_charge,in_charge_slow,in_charge_fast):
            if in_charge_slow==1:
                charging_rate_=[charging_rate[0],charging_rate[1]]
                u_slow= np.maximum(np.minimum(SoC_now+time_in_charge*charging_rate_[0]-80,
                                                                               time_in_charge*charging_rate_[0]),0) 
                charge_gained=time_in_charge*charging_rate_[0]-u_slow+(charging_rate_[1]/charging_rate_[0])*u_slow 
            elif in_charge_fast==1:
                charging_rate_=[charging_rate[2],charging_rate[3]]
                u_slow= np.maximum(np.minimum(SoC_now+time_in_charge*charging_rate_[0]-80,
                                                                               time_in_charge*charging_rate_[0]),0) 
                charge_gained=time_in_charge*charging_rate_[0]-u_slow+(charging_rate_[1]/charging_rate_[0])*u_slow 

            else:
                charge_gained=0

            return SoC_now+charge_gained
        
        for i in range(H):
            fleet_attribute['can_stop_charge_before_'+str(i)]=0
            fleet_attribute.loc[:,'can_stop_charge_before_'+str(i)]=fleet_attribute.apply(lambda row: (get_charge_in_t_min(row['SoC_'],time_step_B*i,row['in_charge_slow'],row['in_charge_fast'])>LB_SoC_for_stop_charging)*
                                                                            (row['in_charge_slow']+row['in_charge_fast']==1),axis=1).astype(int).values


        gdf_fleet_attribute=fleet_attribute[['VehicleId', 'location_']] 
        geometry = [Point(x) for x in gdf_fleet_attribute['location_'].tolist()]
        crs = {'init': 'ptv_mercator'}
        gdf_fleet_attribute = GeoDataFrame(gdf_fleet_attribute, crs=crs, geometry=geometry)

        points_and_zones = sjoin(gdf_fleet_attribute, zones.loc[zones['geometry'].geom_type == 'Polygon'], how="left", op='within')
        fleet_attribute['zone_']=points_and_zones['NO']
        fleet_attribute[['zone_']]=fleet_attribute[['zone_']].fillna(-1)
        fleet_attribute.loc[(fleet_attribute['charger']>0)&
                            ((fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast']+
                              fleet_attribute['charging_planed_slow']+fleet_attribute['charging_planed_fast'])>0)
                            ,'zone_']=pd.merge(fleet_attribute.loc[(fleet_attribute['charger']>0)&
                            ((fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast']+
                              fleet_attribute['charging_planed_slow']+fleet_attribute['charging_planed_fast'])>0),['charger']],
                            charger_attribute[['charger_id','charger_zone']],how='left',left_on='charger',right_on='charger_id')['charger_zone'].values
                                                                              
        performance_update={}
        performance_update['SoC_distribution']=fleet_attribute[['VehicleId','SoC_now']]
        performance_update['SoC_distribution'].rename(columns={'SoC_now':'SoC'},inplace=True)
        files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)
                                                                     
        return fleet_attribute,files_to_track_performance


    def write_tours_and_soc_for_R():
        df_vehicle_tours_and_SoC=fleet_attribute[['VehicleId','SoC_now']]
        df_vehicle_tours_and_SoC=pd.merge(df_vehicle_tours_and_SoC,df_future[['VehicleId','waypoints']],how='left',on='VehicleId')
        path=path_dic['auxilary']
        df_vehicle_tours_and_SoC.to_csv(path+'df_vehicle_tours_and_SoC.csv',sep=';',index=False)

        
    def get_B_input():
        count_for_B_vehicles=fleet_attribute[['zone_','all_vehicles','can_charge','can_trip_0' ]].groupby( ['zone_']).sum().reset_index()

        count_for_B_vehicles_=pd.DataFrame(data={'zone_':np.concatenate([zone_attribute['zone'].unique(),charger_attribute['charger_zone'].unique()],0)})
        count_for_B_vehicles=pd.merge(count_for_B_vehicles_,count_for_B_vehicles,on='zone_',how='left') 
        count_for_B_vehicles=count_for_B_vehicles.fillna(0)
        count_for_B_chargers=fleet_attribute[fleet_attribute['zone_']>=1000][['zone_','in_charge_slow',
                                                                    'in_charge_fast','charging_planed_slow','charging_planed_fast']+
                                            ['should_stop_charging_before_'+str(i) for i in list(range(H))] + ['can_stop_charge_before_'+str(i) for i in list(range(H))]].groupby(
                                            ['zone_']).sum().reset_index()
        import sys
        np.set_printoptions(threshold=sys.maxsize)

        assert len(count_for_B_vehicles)==number_of_all_zones,'count_for_B_vehicles is wrong'

        count_for_B_chargers=pd.merge(charger_zone_attribute,count_for_B_chargers,on='zone_',how='left') 
        count_for_B_chargers=count_for_B_chargers.fillna(0)                                                            
        assert len(count_for_B_chargers)==number_slow_charger_zones+number_fast_charger_zones,'count_for_B_chargers is wrong'                                                         
        count_for_B_vehicles.sort_values(by='zone_',inplace=True)
        count_for_B_chargers.sort_values(by='zone_',inplace=True)

        initial_v0=count_for_B_vehicles['all_vehicles'].values
        path=path_dic['scenario']
        df_B_PUDO=pd.read_csv(path+'df_B_PUDO.csv',sep=';')
        df_B_PUDO_t=df_B_PUDO[df_B_PUDO['time_step'].isin(list(time_of_day_in_hr+np.arange(h)/2))]
        df_B_PUDO_t.sort_values(['time_step','ori'],inplace=True)
        pick_ups_h=np.append(df_B_PUDO_t['trips_PU'].as_matrix(),np.zeros(H*number_of_zones-len(df_B_PUDO_t['trips_PU'])))
        drop_offs_h=np.append(df_B_PUDO_t['trips_DO'].as_matrix(),np.zeros(H*number_of_zones-len(df_B_PUDO_t['trips_PU']))) 
        pick_ups_h=(np.reshape(pick_ups_h,[H,number_of_zones])).astype(int)
        drop_offs_h=(np.reshape(drop_offs_h,[H,number_of_zones])).astype(int)

        path=path_dic['scenario']
        df_cost_matrix=pd.read_csv(path+'df_cost_matrix.csv',sep=';')
        if STAGE==2:
            df_cost_matrix=df_cost_matrix[(df_cost_matrix['origin']<1000)&(df_cost_matrix['destination']<1000)]
        df_cost_matrix['charge']=df_cost_matrix['distance'].apply(lambda x: link_consumption (x))
        cost=np.reshape(df_cost_matrix['charge'].as_matrix(),[number_of_all_zones,number_of_all_zones])
        assert len(set(list(np.reshape(df_cost_matrix['origin'].as_matrix(),[number_of_all_zones,number_of_all_zones])[0])))== 1, 'cost is not shaped correctly'

        can_cut_charging_h=np.transpose(np.array([count_for_B_chargers['can_stop_charge_before_'+str(i)].tolist() for i in list(range(H))]))
        must_cut_charging_h=np.transpose(np.array([count_for_B_chargers['should_stop_charging_before_'+str(i)].tolist() for i in list(range(H))]) )


        avilable_0= count_for_B_vehicles.loc[count_for_B_vehicles['zone_']<1000,'can_trip_0'].values
        chargable_0=count_for_B_vehicles.loc[count_for_B_vehicles['zone_']<1000,'can_charge'].values
        X_h={}
        Y_h={}
        X_h_current={}
        path=path_dic['auxilary']
        with open(path+'num_to_charge.json', "r") as f:
            num_to_charge=json.load(f)
        for type_ in ['slow','fast']:
            Y_h[type_]=np.rint(np.array(DailyPlan['Y_day'][type_][time_step:time_step+H]))
            
            X_h[type_]=np.rint(np.array(DailyPlan['X_day'][type_]))
            X_h[type_]=np.append(np.zeros([0]),X_h[type_],0)
            X_h[type_]=np.cumsum(X_h[type_])
            fp = X_h[type_]
            xp = np.arange(len(X_h[type_]))*time_step_B
            x=time_in_simulation+np.arange(H+1)*time_step_B
            X_h[type_]=np.interp(x, xp, fp)
            X_h[type_]-=num_to_charge[type_]
            X_h[type_]=np.maximum(0,X_h[type_])
            X_h[type_]=np.diff(X_h[type_])
            X_h_current[type_]=np.rint(np.array(DailyPlan['X_day'][type_][time_step:time_step+H]))
            X_h[type_]=np.minimum(1.3*X_h_current[type_],X_h[type_])
            X_h[type_]=np.round(X_h[type_],0)
        return initial_v0,pick_ups_h,drop_offs_h,cost,must_cut_charging_h, can_cut_charging_h, avilable_0, chargable_0,count_for_B_vehicles, X_h, Y_h

       

    def get_C_input(B_answer,files_to_track_performance):
        
        def fromat_B_answer(B_answer,files_to_track_performance):
            B_answer=np.rint(B_answer)
            if number_of_zones_with_charger>0:
                B_answer=B_answer[:-number_of_zones_with_charger*h]
            x_answer_total=B_answer
            slaks={}
            slaks['pick_ups']=B_answer[-h*number_of_zones-6*h:-6*h]
            slaks['in_slow']=B_answer[-6*h:-4*h]
            slaks['in_fast']=B_answer[-4*h:-2*h]
            slaks['go_slow']=B_answer[-2*h:-1*h]
            slaks['go_fast']=B_answer[-1*h:]

            

            performance_update={}
            performance_update['B_slack_going']=pd.DataFrame(data={'slow_in':[slaks['in_slow'][0]],'fast_in':[slaks['in_fast'][0]],
                                                                   'slow_go':[slaks['go_slow'][0]],'fast_go':[slaks['go_fast'][0]],
                                                                   'X_slow':[X_h['slow'][0]],'X_fast':[X_h['fast'][0]], 'chargable':[np.sum(chargable_0)]})
            files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)

            x_answer= B_answer[:-h*number_of_zones-6*h]
            x_3D=np.zeros([h,number_of_zones+number_of_zones_with_charger,number_of_zones+number_of_zones_with_charger])
            x_set=[h*number_of_zones*number_of_zones,h*number_of_zones*number_of_zones+h*number_of_zones*number_of_zones_with_charger]
            x_3D[:,:number_of_zones,:number_of_zones]=np.reshape(np.reshape(x_answer[:x_set[0]],[h,number_of_zones*number_of_zones],'F'),
                                                                 [h,number_of_zones,number_of_zones],'C')
            x_3D[:,:number_of_zones,number_of_zones:]=np.swapaxes(np.reshape(np.reshape(x_answer[x_set[0]:x_set[1]],
                                                                                        [h,number_of_zones*number_of_zones_with_charger],'F'),[h,number_of_zones_with_charger,number_of_zones],'F'),1,2)
            x_3D[:,number_of_zones:,:number_of_zones]=np.reshape(np.reshape(x_answer[x_set[1]:],[h,number_of_zones_with_charger*number_of_zones],'F'),
                                                                 [h,number_of_zones_with_charger,number_of_zones],'F')

            array_zone_id=np.array(count_for_B_vehicles.loc[count_for_B_vehicles['zone_']<1000,'zone_'].tolist())
            array_slow_charger_zone_id=np.array(count_for_B_vehicles.loc[(count_for_B_vehicles['zone_']>=1000)&
                                                             (count_for_B_vehicles['zone_']<10000),'zone_'].tolist())
            array_fast_charger_zone_id=np.array(count_for_B_vehicles.loc[count_for_B_vehicles['zone_']>=10000,'zone_'].tolist())

            mat_flow_relocatin=x_3D[0][:number_of_zones,:number_of_zones]
            mat_flow_charging_1=x_3D[0][:number_of_zones,number_of_zones:number_of_zones+number_slow_charger_zones]
            mat_flow_charging_2=x_3D[0][:number_of_zones,number_of_zones+number_slow_charger_zones:]
            mat_flow_out_of_charging_1=x_3D[0][number_of_zones:number_of_zones+number_slow_charger_zones,:number_of_zones]
            mat_flow_out_of_charging_2=x_3D[0][number_of_zones+number_slow_charger_zones:,:number_of_zones]
            

            def make_df_set_id(origin_list, destination_list, flow, set_id):
                n=len(origin_list)
                m=len(destination_list)
                p=n*m
                o=np.reshape(np.tile(origin_list[:,np.newaxis],[1,m]),[p],'C')
                d=np.reshape(np.tile(destination_list[np.newaxis,:],[n,1]),[p],'C')
                f=np.reshape(flow,[p],'C')
                df=pd.DataFrame({'set_id':set_id*np.ones(p), 'origin':o,'destination':d, 'flow':f})
                return df

            # zone to charger
            ## slow
            df_B_set_1=make_df_set_id(array_zone_id, array_slow_charger_zone_id, mat_flow_charging_1, 11)

            ## fast
            df_B_set_1_=make_df_set_id(array_zone_id, array_fast_charger_zone_id, mat_flow_charging_2, 12)
            df_B_set_1=pd.concat([df_B_set_1,df_B_set_1_])

            # relocation
            df_B_set_2=make_df_set_id(array_zone_id, array_zone_id, mat_flow_relocatin,2)

            # out_of_charger
            ## slow
            df_B_set_3=make_df_set_id(array_slow_charger_zone_id,array_zone_id, mat_flow_out_of_charging_1, 31)

            ## fast
            df_B_set_3_=make_df_set_id( array_fast_charger_zone_id,array_zone_id, mat_flow_out_of_charging_2, 32)
            df_B_set_3=pd.concat([df_B_set_3,df_B_set_3_])

            B_in_charge={'slow':Y_h['slow'][0]-slaks['in_slow'][0]+slaks['in_slow'][h],
                        'fast':Y_h['fast'][0]-slaks['in_fast'][0]+slaks['in_fast'][h]}
##            print('B_in_charge',B_in_charge)
            return df_B_set_1, df_B_set_2, df_B_set_3, B_in_charge





        ## for zone to charger
        #get vehicles and destinations to send to xroute
        def get_potential_vehicles():
            df_set_1_flows=df_B_set_1[(df_B_set_1['flow']>0)]
            list_of_origins_with_posetive_flow=df_set_1_flows['origin'].unique()
            df_set_1_flows_vehicles=fleet_attribute[(fleet_attribute['zone_'].isin(list_of_origins_with_posetive_flow))&
                                                    (fleet_attribute['can_charge']==1)]
            df_set_1_flows_vehicles['zone_']=df_set_1_flows_vehicles['zone_'].astype(int)
            charger_attribute['charger_zone']=charger_attribute['charger_zone'].astype(int)
            df_set_1_flows['origin']=df_set_1_flows['origin'].astype(int)
            df_set_1_flows_vehicles=pd.merge(df_set_1_flows_vehicles, df_set_1_flows ,how='left',left_on='zone_',right_on='origin')
            df_set_1_flows_vehicles=pd.merge(df_set_1_flows_vehicles, charger_attribute,how='left', left_on='destination',right_on='charger_zone')
            avg_EAT=df_set_1_flows_vehicles['EAT'].mean()
            df_set_1_flows_vehicles['flow']=df_set_1_flows_vehicles['flow'].apply(lambda x: np.round(x*(avg_EAT/(0.5*time_step_B))+0.5,0))
    ##        print('ratio going to charge [to c]/[from B]:',avg_EAT/(0.5*time_step_B))
            df_set_1_flows_vehicles=df_set_1_flows_vehicles[['set_id', 'VehicleId','SoC_','location_','EAT','zone_','destination', 'flow', 'charger_id', 'charger_location','capacity','fast','slow']]



            ## for zone to zone (relocation)
            df_set_2_flows=df_B_set_2[(df_B_set_2['flow']>0)]
            list_of_origins_with_posetive_flow=df_set_2_flows['origin'].unique()
            df_set_2_flows_vehicles=fleet_attribute[(fleet_attribute['zone_'].isin(list_of_origins_with_posetive_flow))
                                                                                    &(fleet_attribute['can_trip_0']==1)]
            df_set_2_flows_vehicles=pd.merge(df_set_2_flows_vehicles,df_set_2_flows,how='left',left_on='zone_',right_on='origin')  
            df_set_2_flows_vehicles=pd.merge(df_set_2_flows_vehicles,zone_attribute,how='left',left_on='destination',right_on='zone')
            df_set_2_flows_vehicles=df_set_2_flows_vehicles[['set_id', 'VehicleId','SoC_','location_','EAT','zone_','destination','flow','zone_center_coord']]

            ## for charger_zone to zone
            df_set_3_flows=df_B_set_3[(df_B_set_3['flow']>0)]
            list_of_origins_with_posetive_flow=df_set_3_flows['origin'].unique()
            df_set_3_flows_vehicles=fleet_attribute[(fleet_attribute['zone_'].isin(list_of_origins_with_posetive_flow))
                                                                                    &(fleet_attribute['can_stop_charge_before_0']==1)]

            df_set_3_flows_vehicles=pd.merge(df_set_3_flows_vehicles,df_set_3_flows,how='left',left_on='zone_',right_on='origin')  
            df_set_3_flows_vehicles[['charger','destination']]=df_set_3_flows_vehicles[['charger','destination']].astype(int)#EXTRA
            cost_charger_to_zone_center[['charger','zone']]=cost_charger_to_zone_center[['charger','zone']].astype(int)#EXTRA
            df_set_3_flows_vehicles=pd.merge(df_set_3_flows_vehicles,cost_charger_to_zone_center,how='left',left_on=['charger','destination'],right_on=['charger','zone']) 
            df_set_3_flows_vehicles=df_set_3_flows_vehicles[['set_id', 'VehicleId','SoC_now', 'location_', 'zone_','destination','flow','charger','distance','travel_time']]
            return df_set_1_flows_vehicles,df_set_2_flows_vehicles,df_set_3_flows_vehicles
                 
        ## send requests to xroute
        def get_cost_for_options(df_set_1_flows_vehicles,df_set_2_flows_vehicles,df_set_3_flows_vehicles):
            x_server = XServer()

            def write_distance_and_travel_time_to_df (df, o, d, x_server):   
                def get_travel_time(o,d, x_server):

                    list_coor=[o,d]
                    profile="car"
                    route_request=x_server.make_route_request( profile, list_coor)
                    distance, travel_time, _=x_server.send_route_request(route_request)
                    return [distance, travel_time]
                if len(df)>0:
                    df['dist_TT'] =df[[o,d]].apply(lambda row:get_travel_time(row[o],row[d],x_server), axis=1)   
                    df['travel_time']=df['dist_TT'].apply(lambda x:x[1]) 
                    df['distance']=df['dist_TT'].apply(lambda x:x[0])
                    df.drop(['dist_TT'], axis=1, inplace=True)
                else:
                    df['travel_time']=0
                    df['distance']=0        
                return df

            df_set_1_flows_vehicles=write_distance_and_travel_time_to_df (df_set_1_flows_vehicles, 'location_', 'charger_location', x_server)
            df_set_2_flows_vehicles=write_distance_and_travel_time_to_df (df_set_2_flows_vehicles, 'location_', 'zone_center_coord', x_server)



            df_set_1=df_set_1_flows_vehicles[['set_id' ,'VehicleId','SoC_','EAT','zone_','destination', 'flow', 'charger_id','distance','travel_time']]
            df_set_1.rename(columns={'SoC_':'SoC','zone_':'origin','charger_id':'charger','distance':'cost_distance','travel_time':'cost_time'},inplace=True)
            df_set_1['t']=np.nan
            df_set_1['cost_charge']=df_set_1['cost_distance'].apply(lambda x:link_consumption(x))

            df_set_2=df_set_2_flows_vehicles[['set_id' ,'VehicleId','SoC_','EAT','zone_','destination','flow','distance','travel_time']]
            df_set_2.rename(columns={'SoC_':'SoC','zone_':'origin','distance':'cost_distance','travel_time':'cost_time'},inplace=True)
            df_set_2['t']=np.nan
            df_set_2['charger']=np.nan
            df_set_2['cost_charge']=df_set_2['cost_distance'].apply(lambda x:link_consumption(x))

            df_set_3=df_set_3_flows_vehicles[['set_id' ,'VehicleId','SoC_now', 'zone_','destination','flow','charger','distance','travel_time']]
            df_set_3.rename(columns={'SoC_now':'SoC','zone_':'origin','distance':'cost_distance','travel_time':'cost_time'},inplace=True)
            df_set_3['cost_charge']=df_set_3['cost_distance'].apply(lambda x:link_consumption(x))
            length=len(df_set_3)
            df_set_3=df_set_3.reindex(df_set_3.index.repeat(4))
            df_set_3['t']=np.tile(np.array([0,1,2,3]),length)
            df_set_3['EAT']=0
            def get_charge_in_t_min(SoC_now,time_in_charge,charging_rate):

                u_slow= np.maximum(np.minimum(SoC_now+time_in_charge*charging_rate[0]-80,
                                                                               time_in_charge*charging_rate[0]),0) 
                charge_gained=time_in_charge*charging_rate[0]-u_slow+(charging_rate[1]/charging_rate[0])*u_slow 
                return SoC_now+charge_gained
            df_set_3['time_in_charge']=df_set_3['t'].values*(time_step_B/4)
            if len(df_set_3.loc[df_set_3['set_id']==31])>0:
                df_set_3.loc[df_set_3['set_id']==31,'SoC']=df_set_3.loc[df_set_3['set_id']==31].apply(lambda row:get_charge_in_t_min(row['SoC'],row['time_in_charge'],[charging_rate[0],charging_rate[1]]),axis=1).values
            if len(df_set_3.loc[df_set_3['set_id']==32])>0:
                df_set_3.loc[df_set_3['set_id']==32,'SoC']=df_set_3.loc[df_set_3['set_id']==32].apply(lambda row:get_charge_in_t_min(row['SoC'],row['time_in_charge'],[charging_rate[2],charging_rate[3]]),axis=1).values

            # filter the option id they do not have enough charge to excute it 
            def clean_df(df,col=df_set_1.columns):
                df=df.loc[df['SoC']>df['cost_charge']]
                df=df[col]
                return df
            df_set_1=clean_df(df_set_1)
            df_set_2=clean_df(df_set_2)
            df_set_3=clean_df(df_set_3)
            return df_set_1,df_set_2,df_set_3



        def get_charger_state():
            
            chargers_state=charger_attribute[['charger_id','capacity','charger_zone']]
            chargers_state.rename(columns={'charger_id':'charger'},inplace=True)
            chargers_state_=fleet_attribute[fleet_attribute['zone_']>=1000][['in_charge_slow','in_charge_fast','charger']].groupby(['charger']).sum().reset_index()
            chargers_state['charger']=chargers_state['charger'].astype(int)
            chargers_state_['charger']=chargers_state_['charger'].astype(int)
            chargers_state=pd.merge(chargers_state,chargers_state_,how='left',on='charger')
            chargers_state[['in_charge_slow','in_charge_fast']]=chargers_state[['in_charge_slow','in_charge_fast']].fillna(0)
            out_of_capacity=chargers_state[chargers_state['in_charge_slow']+chargers_state['in_charge_fast']>chargers_state['capacity']]
            if len(out_of_capacity)>0:
                print('charger capacity violation!!!!!')
                print(out_of_capacity)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(fleet_attribute[fleet_attribute['charger']>0])
            df_planed_charging=fleet_attribute[['charging_planed_slow','charging_planed_fast','charger', 'EAT']]

            df_planed_charging.rename(columns={'charger_id':'charger'},inplace=True)
            df_planed_charging=pd.merge(df_planed_charging,charger_attribute[['charger_id','capacity','charger_zone']],how='left',left_on='charger',right_on='charger_id')
            df_planed_charging.rename(columns={'charger_zone':'destination'},inplace=True)

            df_planed_relocation=fleet_attribute[fleet_attribute['relocation_planed']==1][['zone_','relocation_planed']].groupby(['zone_']).sum().reset_index()
            df_planed_relocation.rename(columns={'zone_':'zone'},inplace=True)

            return chargers_state, df_planed_charging, df_planed_relocation
        
        df_B_set_1, df_B_set_2, df_B_set_3,B_in_charge=fromat_B_answer(B_answer,files_to_track_performance)
        df_set_1_flows_vehicles,df_set_2_flows_vehicles,df_set_3_flows_vehicles =get_potential_vehicles() 
        df_set_1,df_set_2,df_set_3=get_cost_for_options(df_set_1_flows_vehicles,df_set_2_flows_vehicles,df_set_3_flows_vehicles)
        chargers_state, df_planed_charging, df_planed_relocation=get_charger_state()
        

        return df_set_1, df_set_2, df_set_3,df_planed_charging,df_planed_relocation,chargers_state,B_in_charge





    def C__get_A_ub (df_set_1, df_set_2, df_set_3, list_of_chargers,
                    df_planed_charging,df_planed_relocation,
                    chargers_state,B_in_charge,UP_SoC_for_charging=50,
                    LB_SoC_for_stop_charging=70,objective_weigths=None,time_bound_step=(time_step_B/4)):
        
        
        set_id_dic={'in':{'slow':11,
                         'fast':12},
                   'out':{'slow':31,
                         'fast':32}}
        df_with_vehicles=pd.concat([df_set_1, df_set_2, df_set_3], axis=0)
        if len(df_with_vehicles)>0:
            df_with_vehicles=df_with_vehicles.reset_index()
            df_with_vehicles['vehicle_action_pair_var_id']=df_with_vehicles.index
            df_with_B_flows=df_with_vehicles[['set_id','origin','destination','flow']].drop_duplicates()
            df_with_B_flows=df_with_B_flows.reset_index()
            df_with_B_flows['flow_pair_var_id']=df_with_B_flows.index
            dic_length_of_var_types={'original': len(df_with_vehicles), 'slak_for_flows': len(df_with_B_flows), 
                                'total_in_charge_slaks_slow':4, 'total_in_charge_slaks_fast':4, 'more_in_charge_slow':4, 'more_in_charge_fast':4}
            num_vars=sum(list(dic_length_of_var_types.values()))
            make_x=lambda x: np.zeros([x,num_vars])
            make_boelian_index=lambda t: [True if i in t else False for i in range(num_vars)]
            dictionaty_A_ub={}
            dictionaty_b_ub={}
            dictionaty_A_eq={}
            dictionaty_b_eq={}
            def make_range(var_type, indexes, dic_length_of_var_types=dic_length_of_var_types):
                preload=0
                if var_type=='original':
                    indexes+=preload
                elif var_type=='slak_for_flows':
                    preload+=dic_length_of_var_types['original']
                    indexes+=preload          
                elif var_type=='total_in_charge_slaks_slow':
                    preload+=dic_length_of_var_types['original']
                    preload+=dic_length_of_var_types['slak_for_flows']
                    indexes+=preload  
                elif var_type=='total_in_charge_slaks_fast':
                    preload+=dic_length_of_var_types['original']
                    preload+=dic_length_of_var_types['slak_for_flows']
                    preload+=dic_length_of_var_types['total_in_charge_slaks_slow']
                    indexes+=preload
                elif var_type=='more_in_charge_slow':
                    preload+=dic_length_of_var_types['original']
                    preload+=dic_length_of_var_types['slak_for_flows']
                    preload+=dic_length_of_var_types['total_in_charge_slaks_slow']
                    preload+=dic_length_of_var_types['total_in_charge_slaks_fast']
                    indexes+=preload
                elif var_type=='more_in_charge_fast':
                    preload+=dic_length_of_var_types['original']
                    preload+=dic_length_of_var_types['slak_for_flows']
                    preload+=dic_length_of_var_types['total_in_charge_slaks_slow']
                    preload+=dic_length_of_var_types['total_in_charge_slaks_fast']
                    preload+=dic_length_of_var_types['more_in_charge_slow']
                    indexes+=preload  
                return indexes

            #flows from B + slak make the flow from B
            df_get_matching_indexs=df_with_vehicles.groupby(by=['set_id','origin','destination'])['vehicle_action_pair_var_id'].apply(list).reset_index()
            df_get_matching_indexs=pd.merge(df_get_matching_indexs,df_with_B_flows,how='left',on=['set_id','origin','destination'])
            df_get_matching_indexs['vehicle_action_pair_var_id']=df_get_matching_indexs['vehicle_action_pair_var_id'].apply(
                                                                                                                    lambda x: np.array(x)).apply(
                                                                                                                    lambda x: make_range('original',
                                                                                                                    x, dic_length_of_var_types))
            df_get_matching_indexs['flow_pair_var_id']=df_get_matching_indexs['flow_pair_var_id'].apply(
                                                                                                lambda x: np.array([x])).apply(
                                                                                                lambda x: make_range('slak_for_flows',
                                                                                                x, dic_length_of_var_types))
            df_get_matching_indexs['indexs_with_value_1_for_costraint_B_flows']=df_get_matching_indexs.apply(
                                                                                lambda row: np.concatenate([row['vehicle_action_pair_var_id'],
                                                                                                            row['flow_pair_var_id']],0),axis=1)

            df_get_matching_indexs['indexs_with_value_1_for_costraint_B_flows']=df_get_matching_indexs['indexs_with_value_1_for_costraint_B_flows'].apply(make_boelian_index)

            indexes_for_A_ub=np.array(df_get_matching_indexs['indexs_with_value_1_for_costraint_B_flows'].tolist())

            A_ub=make_x(len(df_get_matching_indexs))
            A_ub[indexes_for_A_ub]=1
            b_ub=df_get_matching_indexs.as_matrix(columns=['flow'])[:,0]
            dictionaty_A_eq['satisfy_demand']=csr_matrix(A_ub)
            dictionaty_b_eq['satisfy_demand']=b_ub 
            #charger capacity should not exceed  for each t

            for t in range(4):
                df_set_1=df_with_vehicles[df_with_vehicles['set_id'].isin([11,12])]
                df_set_3=df_with_vehicles[df_with_vehicles['set_id'].isin([31,32])]
                

                df_set_1=df_set_1[df_set_1['EAT']+df_set_1['cost_time']<= (t+1)*time_bound_step]
                df_set_3=df_set_3[df_set_3['t']<=t]

                df_planed_charging_t=df_planed_charging[df_planed_charging['EAT']<=(t+1)*time_bound_step]
                df_planed_charging_t['flow_preload']=df_planed_charging_t['charging_planed_slow']+df_planed_charging_t['charging_planed_fast']
                df_planed_charging_agg=df_planed_charging_t.groupby(['charger']).sum().reset_index()



                df_get_matching_indexs=pd.DataFrame(df_set_1.groupby(['charger'])['vehicle_action_pair_var_id'].apply(list)).reset_index()
                df_get_matching_3_indexs=pd.DataFrame(df_set_3.groupby(['charger'])['vehicle_action_pair_var_id'].apply(list)).reset_index()

                df_get_matching_indexs=pd.merge(df_get_matching_indexs,df_get_matching_3_indexs,how='outer',on=['charger'],suffixes=('_in','_out') )

                for name in ['vehicle_action_pair_var_id_in', 'vehicle_action_pair_var_id_out']:
                    df_get_matching_indexs[name] = df_get_matching_indexs[name].apply(lambda d: d if isinstance(d, list) else [])

                df_get_matching_indexs=pd.merge(df_get_matching_indexs,chargers_state,how='left',on=['charger'])
                df_get_matching_indexs=pd.merge(df_get_matching_indexs,df_planed_charging_agg[['charger','flow_preload']],how='left',on=['charger'])
                df_get_matching_indexs[['flow_preload']]=df_get_matching_indexs[['flow_preload']].fillna(0)
                df_get_matching_indexs['vehicle_action_pair_var_id_in']=df_get_matching_indexs['vehicle_action_pair_var_id_in'].apply(
                                                                                                                        lambda x: np.array(x)).apply(
                                                                                                                        lambda x: make_range('original',
                                                                                                                        x, dic_length_of_var_types))
                df_get_matching_indexs['vehicle_action_pair_var_id_out']=df_get_matching_indexs['vehicle_action_pair_var_id_out'].apply(
                                                                                                                        lambda x: np.array(x)).apply(
                                                                                                                        lambda x: make_range('original',
                                                                                                                        x, dic_length_of_var_types))
                if len(df_get_matching_indexs)>0:
                    df_get_matching_indexs['vehicle_action_pair_var_id_in']=df_get_matching_indexs['vehicle_action_pair_var_id_in'].apply(make_boelian_index)
                    indexes_for_A_ub=np.array(df_get_matching_indexs['vehicle_action_pair_var_id_in'].tolist())
                    A_ub=make_x(len(df_get_matching_indexs))
                    A_ub[indexes_for_A_ub]=1

                    df_get_matching_indexs['vehicle_action_pair_var_id_out']=df_get_matching_indexs['vehicle_action_pair_var_id_out'].apply(make_boelian_index)
                    indexes_for_A_ub=np.array(df_get_matching_indexs['vehicle_action_pair_var_id_out'].tolist())
                    A_ub[indexes_for_A_ub]=-1

                    df_get_matching_indexs['charger_capacity']=df_get_matching_indexs['capacity']-(df_get_matching_indexs['flow_preload']+
                                                                                                  df_get_matching_indexs['in_charge_slow']+
                                                                                                  df_get_matching_indexs['in_charge_fast'])

                    b_ub=df_get_matching_indexs['charger_capacity'].values
                    dictionaty_A_ub['charger_capacity_'+str(t)]=csr_matrix(A_ub)
                    dictionaty_b_ub['charger_capacity_'+str(t)]=b_ub  


            #each vehicle can only fulfil one destination
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


    def C__solve_A_ub (S_ub, b_ub, S_eq, b_eq, c, dic_length_of_var_types, gap=0.01):
        model = gurobipy.Model()
        model.setParam("OutputFlag",0)
        rows_ub, rows_eq, cols = len(b_ub),len(b_eq), len(c)
        x_answer=np.zeros([cols])
        x = []

        for j in range(dic_length_of_var_types['original']):
            x.append(model.addVar( obj=c[j], vtype=gurobipy.GRB.BINARY))
        for j in range(len(x),cols):
            x.append(model.addVar(lb=0, obj=c[j]))

        model.update()

        for i in range(rows_ub):
          start = S_ub.indptr[i]
          end   = S_ub.indptr[i+1]
          variables = [x[j] for j in S_ub.indices[start:end]]
          coeff     = S_ub.data[start:end]
          expr = gurobipy.LinExpr(coeff, variables)
          model.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=b_ub[i])
        model.update() 
        

        for i in range(rows_eq):
          start = S_eq.indptr[i]
          end   = S_eq.indptr[i+1]
          variables = [x[j] for j in S_eq.indices[start:end]]
          coeff     = S_eq.data[start:end]
          expr = gurobipy.LinExpr(coeff, variables)
          model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b_eq[i])
        model.update() 

        model.ModelSense = gurobipy.GRB.MINIMIZE
        model.setParam("MIPGap", gap)
        model.setParam('TimeLimit',60)
        model.update()
        model.optimize()
        
        i=0
        if (model.status == gurobipy.GRB.Status.OPTIMAL) or (model.status == gurobipy.GRB.Status.TIME_LIMIT):
            i=0
            for v in model.getVars():
                x_answer[i]=v.x
                i+=1

            start = time.time()
            return x_answer, 1
        else:
            print('status is:',model.status, 'returing model instead' )
            return model, 0



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
            
            
            #update performance files
            performance_update={}
            performance_update['cost_to_charger']=df_with_vehicles.loc[df_with_vehicles['set_id'].isin([11,12]),
                                                                       ['VehicleId','charger','cost_charge','cost_time','cost_distance','SoC']]
            
            performance_update['cost_relocation']=df_with_vehicles.loc[df_with_vehicles['set_id']==2,
                                                                       ['VehicleId','origin','destination','cost_charge','cost_time','cost_distance','SoC']]
            performance_update['cost_relocation'].rename(columns={'destination':'to_zone','origin':'from_zone'},inplace=True)
            
            performance_update['cost_from_charger']=df_with_vehicles.loc[df_with_vehicles['set_id'].isin([31,32]),
                                                                       ['VehicleId','charger','destination','cost_charge','cost_time','cost_distance','SoC']]
            performance_update['cost_from_charger'].rename(columns={'destination':'to_zone'},inplace=True)

            performance_update['C_slack']=pd.DataFrame({'flow_slack':[s_1],
                                                        'in_charge_slow':[s_2],
                                                        'in_charge_fast':[s_3]})
            files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)
            return df_set_1, df_set_2_3,files_to_track_performance
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






    if LOG==1:
        print('charging script: reading files')

    path=path_dic['auxilary']
    with open(path+'time_in_simulation.txt', "r") as f:
        time_in_simulation=int(f.read())

    path=path_dic['from_simulator']
    with open('EVehicleChargingDeletionRequests.json', "w") as f:
        json.dump
    with open('EVehicleChargingRequests.json', "w") as f:
        json.dump
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

            
        if LOG==1:
            print('charging script: B')
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


        if B_success==0:
            B_answer, B_success=stage_B__solve_gurobi(S_ub, b_ub,S_eq, b_eq, c, 60)
            if B_success==0:
                print('charging script: B is not feasible. Returning the input to B')
                import sys
                np.set_printoptions(threshold=sys.maxsize)
                for name in ['h', 'number_of_zones', 'number_chargers_in_zones_with_charger',
                                     'number_slow_charger_zones', 'number_fast_charger_zones', 
                                     'initial_v0','pick_ups_h','drop_offs_h' , 
                                     'X_h','Y_h','must_cut_charging_h', 'can_cut_charging_h', 
                                     'avilable_0', 'chargable_0']:
                    print(name)
                    print(eval(name))
                fleet_attribute.to_csv('fleet_attribute_check_for_errors.csv',sep=';',index=False)
            

            else:
                print('charging script: |WARN| B took long to solve. Consider reducing number of zones')
        if LOG==1:
            print('charging script: C')
        df_set_1, df_set_2, df_set_3,df_planed_charging,df_planed_relocation,chargers_state,B_in_charge=get_C_input(B_answer,files_to_track_performance)

        performance_update={}
        performance_update['charger_occupation']=chargers_state[['charger','in_charge_slow','in_charge_fast']]
        performance_update['charger_occupation']['num']=performance_update['charger_occupation']['in_charge_slow']+performance_update['charger_occupation']['in_charge_fast']
        files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)
                
        list_from_C,posetive_flow_from_B=C__get_A_ub (df_set_1, df_set_2, df_set_3,
                                                                    list_of_chargers, 
                                                                    df_planed_charging,df_planed_relocation,
                                                                    chargers_state,B_in_charge,UP_SoC_for_charging=50,
                                                                    LB_SoC_for_stop_charging=70,objective_weigths=penalty_c,time_bound_step=(time_step_B/4))

        if posetive_flow_from_B==1:
            [S_ub, b_ub, S_eq, b_eq, c, dictionaty_A_ub, dictionaty_b_ub, dictionaty_A_eq, dictionaty_b_eq, dic_length_of_var_types, df_with_vehicles]=list_from_C
            C_answer, C_success=C__solve_A_ub (S_ub, b_ub, S_eq, b_eq, c, dic_length_of_var_types, gap=0.01)
            if C_success==0:
                print('charging script: C is not feasible. Returning the input to C')
                import sys
                np.set_printoptions(threshold=sys.maxsize)
                for name in ['df_set_1', 'df_set_2', 'df_set_3','list_of_chargers', 
                                                                    'df_planed_charging','df_planed_relocation',
                                                                    'chargers_state','B_in_charge']:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(name)
                        print(eval(name))

                fleet_attribute.to_csv('fleet_attribute_check_for_errors.csv',sep=';',index=False)
            if LOG==1:
                print('charging script: write requests')
            apply_actions(C_answer, df_with_vehicles,files_to_track_performance,fleet_attribute,dic_length_of_var_types)
        else:
            if LOG==1:
                print('charging script: no requests')
            apply_actions([], [],files_to_track_performance,fleet_attribute,[], code=0)

    path=path_dic['auxilary']

    open(path+'time_in_simulation.txt', 'w').close()
    if LOG==1:
        print('charging script: time ',time_in_simulation)
    with open(path+'time_in_simulation.txt', "w") as f:
        f.write(str(time_in_simulation))
    if LOG==1:
        print('charging script: im done')
if lazy_charging:

    def make_fleet_attribute(VehicleId_list):
        zeros=np.zeros([feet_size])
        ones=np.ones([feet_size])
        fleet_attribute=pd.DataFrame(data={'VehicleId':VehicleId_list,
                                           'SoC_now':100*ones,
                                           'charging_planed_slow':zeros,
                                           'charging_planed_fast':zeros,
                                           'Expected_start_charging_time':zeros,
                                           'in_charge_slow':zeros,
                                           'in_charge_fast':zeros,
                                           'charger':zeros,
                                           'charge_duration':zeros,
                                           'trip_id':zeros,
                                           'last_row_trejectory':zeros})
        return fleet_attribute
    def make_files_to_track_performance(files_to_track_performance,task,to_append={}):

        path=path_dic['results']
        for name in list(files_to_track_performance.keys()):
            if task=='make':
                files_to_track_performance[name]['df']=pd.DataFrame(columns=files_to_track_performance[name]['columns'])
                files_to_track_performance[name]['df'].to_csv(path+name+'.csv',sep=';',index=False)
            if task=='read':
                files_to_track_performance[name]['df']=pd.read_csv(path+name+'.csv',sep=';')
            if task=='write':
                with open(path+name+'.csv', 'a') as f:
                    if len(files_to_track_performance[name]['df'])>0:
                        files_to_track_performance[name]['df'].to_csv(f, header=False, index=False,sep=';')
        if task=='update':
            for name in list(to_append.keys()):
                to_append[name]['time']=time_in_simulation
                to_append[name]=to_append[name][files_to_track_performance[name]['columns']]
                files_to_track_performance[name]['df']=to_append[name]
        return files_to_track_performance

    def read_static_data():
        path=path_dic['scenario']
        charger_attribute=pd.read_csv(path+'charger_attribute.csv',sep=';')
        charger_attribute['charger_location']=charger_attribute['charger_location'].apply(lambda x:literal_eval(x))
        zone_attribute=pd.read_csv(path+'zone_attribute.csv',sep=';')
        zone_attribute['zone_center_coord']=zone_attribute['zone_center_coord'].apply(lambda x:literal_eval(x))
        cost_charger_to_zone_center=pd.read_csv(path+'cost_charger_to_zone_center.csv',sep=';')
        df=pd.read_csv(path+"Zones_ptv_mercator.csv",sep=';')
        df['PTV_mercator_list']=df['PTV_mercator_list'].apply(lambda x:literal_eval(x))
        geometry = [Polygon(x) for x in df.PTV_mercator_list.tolist()]
        crs = {'init': 'ptv_mercator'}
        zones = GeoDataFrame(df, crs=crs, geometry=geometry)
        list_of_chargers={'slow':charger_attribute.loc[charger_attribute['slow']==1,'charger_id'].tolist(),
                          'fast':charger_attribute.loc[charger_attribute['fast']==1,'charger_id'].tolist()}

        closest_charger_zones=pd.read_pickle(path+"closest_charger_zones.pkl")
        #zone counts
        charger_zone_attribute=charger_attribute.groupby(['charger_zone'])['capacity'].sum().reset_index().sort_values(by='charger_zone')
        charger_zone_attribute.rename(columns={'charger_zone':'zone_'},inplace=True)
        number_of_zones=len(zone_attribute)
        number_chargers_in_slow_charger_zones=charger_zone_attribute[charger_zone_attribute['zone_']<10000].as_matrix(
                                                                                        columns=['capacity'])[:,0]
        number_chargers_in_fast_charger_zones=charger_zone_attribute[charger_zone_attribute['zone_']>=10000].as_matrix(
                                                                                        columns=['capacity'])[:,0]
        number_chargers_in_zones_with_charger=np.concatenate([number_chargers_in_slow_charger_zones,
                                                              number_chargers_in_fast_charger_zones],0)
        number_slow_charger_zones=len(charger_zone_attribute[charger_zone_attribute['zone_']<10000])
        number_fast_charger_zones=len(charger_zone_attribute[charger_zone_attribute['zone_']>=10000])

        number_of_zones_with_charger=number_slow_charger_zones+number_fast_charger_zones
        number_of_all_zones=number_of_zones+number_slow_charger_zones+number_fast_charger_zones
        return (charger_attribute,charger_zone_attribute,number_chargers_in_zones_with_charger,
                number_slow_charger_zones,number_fast_charger_zones,number_of_zones_with_charger,
                zone_attribute,number_of_zones,number_of_all_zones,
                cost_charger_to_zone_center, zones, list_of_chargers,closest_charger_zones)

    def update_fleet_attribute_with_simulation_data(fleet_attribute,files_to_track_performance):
        fleet_attribute=pd.merge(fleet_attribute, df_past,how='left',on='VehicleId')
        fleet_attribute=pd.merge(fleet_attribute, df_future,how='left',on='VehicleId')
        fleet_attribute['last_row_trejectory']+=fleet_attribute['number_of_rows']
        fleet_attribute['SoC_now']=fleet_attribute['SoC_now']-fleet_attribute['consumption_past']

        fleet_attribute.drop(['consumption_past','consumption_future'],axis=1)
        ## increased by charging
        fleet_attribute['time_in_charge']=time_step_online*(fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast'])
        fleet_attribute['time_in_charge']+=fleet_attribute.apply(lambda row: np.maximum(time_in_simulation-
                                                                                        row['Expected_start_charging_time'],0)*
                                                                 (row['charging_planed_slow']+row['charging_planed_fast']),axis=1)
        fleet_attribute['u_slow']=fleet_attribute.apply(lambda  row: np.maximum(np.minimum(row['SoC_now']+row['in_charge_slow']*
                                                                                row['time_in_charge']*charging_rate[0]-80,
                                                                       row['in_charge_slow']*row['time_in_charge']*charging_rate[0]),0) ,axis=1)
        fleet_attribute['charge_gained']=fleet_attribute.apply(lambda  row: (row['time_in_charge']*charging_rate[0]-row['u_slow']+
                                                                             (charging_rate[1]/charging_rate[0])*row['u_slow'])
                                                               *row['in_charge_slow'],axis=1)
        charge_gained_slow=fleet_attribute['charge_gained'].sum()
        fleet_attribute['u_fast']=fleet_attribute.apply(lambda  row: np.maximum(np.minimum(row['SoC_now']+row['in_charge_fast']*
                                                                                row['time_in_charge']*charging_rate[2]-80,
                                                                       row['in_charge_fast']*row['time_in_charge']*
                                                                                charging_rate[2]),0),axis=1)
        fleet_attribute['charge_gained']+=fleet_attribute.apply(lambda  row: (row['time_in_charge']*charging_rate[2]-row['u_fast']+
                                                                              (charging_rate[3]/charging_rate[2])*row['u_fast'])
                                                                *row['in_charge_fast'],axis=1)
        charge_gained_fast=fleet_attribute['charge_gained'].sum()-charge_gained_slow
        performance_update={}
        performance_update['charge_gained']=pd.DataFrame(data={'slow':[charge_gained_slow],'fast':[charge_gained_fast]})
        files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)

        # Update state of in charge
        fleet_attribute['in_charge_slow']+=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_slow'],axis=1)
        fleet_attribute['charging_planed_slow']-=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_slow'],axis=1)\


        fleet_attribute['in_charge_fast']+=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_fast'],axis=1)
        fleet_attribute['charging_planed_fast']-=fleet_attribute.apply(lambda row:(time_in_simulation-
                                                                        row['Expected_start_charging_time']>0)*
                                                                        row['charging_planed_fast'],axis=1)



        fleet_attribute.loc[(fleet_attribute['charging_planed_slow']==0)&(fleet_attribute['charging_planed_fast']==0),'Expected_start_charging_time']=0
        fleet_attribute.loc[(fleet_attribute['charging_planed_slow']==0)&(fleet_attribute['charging_planed_fast']==0)&
                            (fleet_attribute['in_charge_slow']==0)&(fleet_attribute['in_charge_fast']==0),'charger']=0
        fleet_attribute.loc[((fleet_attribute['charging_planed_slow']==1)+
                            (fleet_attribute['charging_planed_fast']==1)+
                            (fleet_attribute['in_charge_slow']==1)+
                            (fleet_attribute['in_charge_fast']==1))==0,'trip_id']=0

        fleet_attribute['SoC_now']=fleet_attribute['SoC_now']+fleet_attribute['charge_gained']
        fleet_attribute['SoC_now']=fleet_attribute['SoC_now'].apply(lambda x:min(100,x))
        fleet_attribute['SoC_']=fleet_attribute['SoC_now']-fleet_attribute['consumption_future']

        #check for faults
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['in_charge_fast']).max()<=1, 'in charge not right'
        assert  (fleet_attribute['charging_planed_fast']+ fleet_attribute['charging_planed_slow']).max()<=1, 'planned value not right'
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['charging_planed_slow']).max()<=1, 'in charge or planned value not right'
        if fleet_attribute['SoC_now'].max()>100:
            fleet_attribute.to_csv('fleet_attribute_check_for_errors.csv',sep=';',index=False)
            assert  fleet_attribute['SoC_now'].max()<=100, 'SoC over 100'
        if fleet_attribute['SoC_now'].min()<0:
            print('following vehicles have SoC below 0: ',fleet_attribute.loc[fleet_attribute['SoC_now']<0,'VehicleId'].tolist())



        #stop charging
        fleet_attribute.loc[fleet_attribute['SoC_now']>stop_charge_limit,[
                                 'in_charge_slow',
                                 'in_charge_fast',
                                 'charger',
                                 'charge_duration',
                                'trip_id']]=0

        #check for faults
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['in_charge_fast']).max()<=1, 'in charge not right'
        assert  (fleet_attribute['charging_planed_fast']+ fleet_attribute['charging_planed_slow']).max()<=1, 'planned value not right'
        assert  (fleet_attribute['in_charge_slow']+ fleet_attribute['charging_planed_slow']).max()<=1, 'in charge or planned value not right'



        gdf_fleet_attribute=fleet_attribute[['VehicleId', 'location_']]
        geometry = [Point(x) for x in gdf_fleet_attribute['location_'].tolist()]
        crs = {'init': 'ptv_mercator'}
        gdf_fleet_attribute = GeoDataFrame(gdf_fleet_attribute, crs=crs, geometry=geometry)

        points_and_zones = sjoin(gdf_fleet_attribute, zones.loc[zones['geometry'].geom_type == 'Polygon'], how="left", op='within')
        fleet_attribute['zone_']=points_and_zones['NO']
        fleet_attribute[['zone_']]=fleet_attribute[['zone_']].fillna(-1)
        fleet_attribute.loc[(fleet_attribute['charger']>0)&
                            ((fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast'])>0)
                            ,'zone_']=pd.merge(fleet_attribute.loc[(fleet_attribute['charger']>0)&
                            ((fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast'])>0),['charger']],
                            charger_attribute[['charger_id','charger_zone']],how='left',left_on='charger',right_on='charger_id')['charger_zone'].values
        performance_update={}
        performance_update['SoC_distribution']=fleet_attribute[['VehicleId','SoC_now']]
        performance_update['SoC_distribution'].rename(columns={'SoC_now':'SoC'},inplace=True)
        files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)

        return fleet_attribute, gdf_fleet_attribute, files_to_track_performance



    def write_tours_and_soc_for_R():
        df_vehicle_tours_and_SoC=fleet_attribute[['VehicleId','SoC_now']]
        df_vehicle_tours_and_SoC=pd.merge(df_vehicle_tours_and_SoC,df_future[['VehicleId','waypoints']],how='left',on='VehicleId')
        path=path_dic['auxilary']
        df_vehicle_tours_and_SoC.to_csv(path+'df_vehicle_tours_and_SoC.csv',sep=';',index=False)


    def get_charger_state():

        chargers_state=charger_attribute[['charger_id','capacity','charger_zone']]
        chargers_state.rename(columns={'charger_id':'charger'},inplace=True)
        chargers_state_=fleet_attribute[fleet_attribute['zone_']>=1000][['in_charge_slow','in_charge_fast','charger']].groupby(['charger']).sum().reset_index()
        chargers_state['charger']=chargers_state['charger'].astype(int)
        chargers_state_['charger']=chargers_state_['charger'].astype(int)
        chargers_state=pd.merge(chargers_state,chargers_state_,how='left',on='charger')
        chargers_state[['in_charge_slow','in_charge_fast']]=chargers_state[['in_charge_slow','in_charge_fast']].fillna(0)
        return chargers_state


    def lazy_charging(fleet_attribute):

        fleet_attribute['must_charge']=0
        fleet_attribute.loc[(fleet_attribute['SoC_']<soc_must_charge)&
                        (fleet_attribute['charging_planed_slow']+fleet_attribute['charging_planed_fast']+
                         fleet_attribute['in_charge_slow']+fleet_attribute['in_charge_fast']==0),'must_charge']=1
        fleet_attribute['potential_charger']=fleet_attribute['charger']
        points_and_zones = sjoin(gdf_fleet_attribute, closest_charger_zones, how="left", op='within')[['VehicleId','charger_id']]
        fleet_attribute.loc[fleet_attribute['must_charge']>0,'potential_charger']=pd.merge(
            fleet_attribute.loc[fleet_attribute['must_charge']>0],points_and_zones,how='left',on='VehicleId')['charger_id'].values

        fleet_attribute.loc[(fleet_attribute['potential_charger'].isnull())&(fleet_attribute['must_charge']>0),'potential_charger']=charger_attribute[charger_attribute['capacity']==charger_attribute['capacity'].max()]['charger_id'].item()
        count_for_B_chargers=fleet_attribute[fleet_attribute['potential_charger']>0][['potential_charger','in_charge_slow',
                                              'in_charge_fast','charging_planed_slow','charging_planed_fast','must_charge']].groupby(
                                            ['potential_charger']).sum().reset_index()
        fleet_attribute=pd.merge(fleet_attribute,charger_attribute,how='left',left_on='potential_charger',right_on='charger_id')

        count_for_B_chargers=pd.merge(count_for_B_chargers,charger_attribute,how='left',left_on='potential_charger',right_on='charger_id')
        count_for_B_chargers=count_for_B_chargers.fillna(0)
        count_for_B_chargers['can_add_to_charger']=count_for_B_chargers['capacity']-(count_for_B_chargers['in_charge_slow']+
                                                                                     count_for_B_chargers['in_charge_fast']+
                                                                                     count_for_B_chargers['charging_planed_slow']+
                                                                                     count_for_B_chargers['charging_planed_fast'])
        count_for_B_chargers['must_choose']=count_for_B_chargers['can_add_to_charger']<count_for_B_chargers['must_charge']
        fleet_attribute_must_charge=fleet_attribute[fleet_attribute['must_charge']>0]
        fleet_attribute['cost_time']=0
        fleet_attribute['cost_distance']=0
        def write_distance_and_travel_time_to_df (df, o, d, x_server):
            def get_travel_time(o,d, x_server):

                list_coor=[o,d]
                profile="car"
                route_request=x_server.make_route_request( profile, list_coor)
                distance, travel_time, _=x_server.send_route_request(route_request)
                return [distance, travel_time]
            if len(df)>0:
                df['dist_TT'] =df[[o,d]].apply(lambda row:get_travel_time(row[o],row[d],x_server), axis=1)
                df['travel_time']=df['dist_TT'].apply(lambda x:x[0])
                df['distance']=df['dist_TT'].apply(lambda x:x[1])
                df.drop(['dist_TT'], axis=1, inplace=True)
            else:
                df['travel_time']=0
                df['distance']=0
            return df
        x_server = XServer()
        fleet_attribute_must_charge=write_distance_and_travel_time_to_df (fleet_attribute_must_charge, 'location_', 'charger_location', x_server)
        fleet_attribute.loc[fleet_attribute['must_charge']>0,'cost_time']=fleet_attribute_must_charge['travel_time'].values
        fleet_attribute.loc[fleet_attribute['must_charge']>0,'cost_distance']=fleet_attribute_must_charge['distance'].values
        fleet_attribute['cost_charge']=fleet_attribute['cost_distance'].apply(lambda x:link_consumption(x))
        df_list_of_distance=fleet_attribute[(fleet_attribute['potential_charger']>0)&(fleet_attribute['must_charge']==True)].groupby(['potential_charger'])['cost_distance'].apply(list).reset_index()
        df_list_of_distance.rename(columns={'cost_distance':'distance_list'},inplace=True)
        fleet_attribute=pd.merge(fleet_attribute,df_list_of_distance,how='left',on='potential_charger')
        fleet_attribute['send_charge']=0
        fleet_attribute=pd.merge(fleet_attribute,count_for_B_chargers[['potential_charger','can_add_to_charger','must_choose']],
                                how='left', on='potential_charger')
        fleet_attribute['send_charge']=0
        df_charge=fleet_attribute[(fleet_attribute['must_choose']==1)&(fleet_attribute['must_charge']==1)&(fleet_attribute['can_add_to_charger']>0)]

        if len(df_charge)>0:
            df_charge.loc[(df_charge['cost_distance']<=df_charge[['distance_list',
                                 'can_add_to_charger']].apply(lambda row:sorted(row['distance_list']+list(np.repeat(np.array(20000),10)))[int(row['can_add_to_charger']-1)],axis=1)),'send_charge']=1
            fleet_attribute.loc[(fleet_attribute['must_choose']==1)&(fleet_attribute['must_charge']==1)&(fleet_attribute['can_add_to_charger']>0),'send_charge']=df_charge['send_charge'].values
        fleet_attribute.loc[(fleet_attribute['must_choose']==0)&
                            (fleet_attribute['must_charge']==1),'send_charge']=1




        fleet_attribute['duration']=0
        fleet_attribute.loc[(fleet_attribute['send_charge']>0),'charger']=fleet_attribute.loc[(fleet_attribute['send_charge']>0),
                                                                                              'potential_charger']



        def get_duration_of_charging(send_charge,SoC_,cost_charge,charger):
            if charger in list_of_chargers['slow']:
                rate=charging_rate[:2]
            if charger in list_of_chargers['fast']:
                rate=charging_rate[2:]
            SoC_start=SoC_-cost_charge
            t1=(80-SoC_start)/rate[0]
            t2=(stop_charge_limit-80)/rate[1]
            return t1+t2
                
        fleet_attribute.loc[fleet_attribute['send_charge']==1,'duration']=fleet_attribute.loc[fleet_attribute['send_charge']==1].apply(
                                                                          lambda row: get_duration_of_charging(row['send_charge'],row['SoC_'],row['cost_charge'],row['charger']),axis=1)

        fleet_attribute.loc[fleet_attribute['send_charge']>0,
                       'Expected_start_charging_time']=(fleet_attribute.loc[fleet_attribute['send_charge']>0,'EAT'].values+
                        fleet_attribute.loc[fleet_attribute['send_charge']>0,'cost_time'].values+time_in_simulation)

        fleet_attribute.loc[(fleet_attribute['send_charge']>0)&
                            (fleet_attribute['charger'].isin(list_of_chargers['slow'])),
                       'charging_planed_slow']=1
        return fleet_attribute

    def apply_actions(files_to_track_performance,fleet_attribute, code=1):
        def get_posetive_actions_form_c (files_to_track_performance):

            df_set_1=fleet_attribute.loc[fleet_attribute['send_charge']>0,['VehicleId','charger','cost_time','duration']]
            df_set_1=pd.merge(df_set_1,charger_attribute[['charger_id', 'charger_location']],how='left',left_on='charger',right_on='charger_id')


            #update performance files
            performance_update={}
            performance_update['cost_to_charger']=fleet_attribute.loc[fleet_attribute['send_charge']>0,
                                                                        ['VehicleId','charger','cost_charge','cost_time','cost_distance','SoC_']]
            performance_update['cost_to_charger'].rename(columns={'SoC_': 'SoC'}, inplace=True)   
            files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)
            return df_set_1,files_to_track_performance
        def make_request_column(df_set_1):
            def make_charge_relocate_request(TripId, coord, VehicleId, time, time_now):
                request={"TripId":TripId,
                         "Time":time_now,
                         "TargetServiceTime":{"Seconds":time,"Nanos":0},
                         "TargetLocation":{"X":coord[0],"Y":coord[1]},
                         "VehicleId":VehicleId}
                return request


            last_used_id={}
            time_now={"Seconds":time_in_simulation,"Nanos":0}
            path=path_dic['auxilary']
            for id_ in ['charge']:
                with open(path+'last_used_id_'+id_+'.txt', "r") as f:
                    last_used_id[id_]=int(f.read())
                open(path+'last_used_id_'+id_+'.txt', 'w').close()

            df_set_1.index = range(1,len(df_set_1.index)+1)
            df_set_1['id']=(df_set_1.index+last_used_id['charge']).astype(int)
            last_used_id['charge']+=len(df_set_1.index)


            if len(df_set_1)>0:
                df_set_1['json_format']=df_set_1.apply(
                                            lambda row: make_charge_relocate_request(row['id'], row['charger_location'],
                                                                                     row['VehicleId'],row['duration'],time_now), axis=1)
                print('send to charger:',df_set_1['VehicleId'].tolist())
            else:
                df_set_1['json_format']=0


            path=path_dic['auxilary']
            for id_ in ['charge']:
                with open(path+'last_used_id_'+id_+'.txt', "w") as f:
                    f.write(str(last_used_id[id_]))
            return df_set_1

        def write_requests_to_file():
            path=path_dic['from_simulator']
            with open(path+'EVehicleChargingRequests.json', "w") as f:
                request_list=[]
                if len(df_set_1)>0:
                    request_list+=df_set_1['json_format'].tolist()
                json.dump(request_list, f, default=str)




        def update_fleet_attribute(fleet_attribute,df_set_1):
            #deleteif charging duration is over
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


            
            return fleet_attribute
        def write_fleet_attribute(fleet_attribute):
            fleet_attribute=fleet_attribute[['VehicleId',
                                           'SoC_now',
                                           'charging_planed_slow',
                                           'charging_planed_fast',
                                           'Expected_start_charging_time',
                                           'in_charge_slow',
                                           'in_charge_fast',
                                           'charger',
                                           'charge_duration',
                                           'trip_id',
                                             'last_row_trejectory']]
            path=path_dic['auxilary']
            fleet_attribute.to_csv(path+'fleet_attribute.csv',sep=';',index=False)

        if code==1:
            df_set_1,files_to_track_performance=get_posetive_actions_form_c(files_to_track_performance)
            df_set_1=make_request_column(df_set_1)
            write_requests_to_file()
            fleet_attribute=update_fleet_attribute(fleet_attribute,df_set_1)
        write_fleet_attribute(fleet_attribute)
        _=make_files_to_track_performance(files_to_track_performance,'write')


    if LOG==1:
        print('charging script: reading files')

    path=path_dic['auxilary']
    with open(path+'time_in_simulation.txt', "r") as f:
        time_in_simulation=int(f.read())

    path=path_dic['from_simulator']
    with open('EVehicleChargingDeletionRequests.json', "w") as f:
        json.dump
    with open('EVehicleChargingRequests.json', "w") as f:
        json.dump
    time_in_simulation+=time_step_online
    time_of_day_in_hr=operation_start_time+int(time_in_simulation/(30*60))/2
    time_step=int(time_in_simulation/(30*60))
    continue_script=record_tours()




    if continue_script==1:        
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
    cost_charger_to_zone_center, zones, list_of_chargers,closest_charger_zones)=read_static_data()

    fleet_attribute, gdf_fleet_attribute, files_to_track_performance=update_fleet_attribute_with_simulation_data(fleet_attribute,files_to_track_performance)
    write_tours_and_soc_for_R()
    chargers_state=get_charger_state()
    
    fleet_attribute=lazy_charging(fleet_attribute)
    
    performance_update={}
    performance_update['charger_occupation']=chargers_state[['charger','in_charge_slow','in_charge_fast']]
    performance_update['charger_occupation']['num']=performance_update['charger_occupation']['in_charge_slow']+performance_update['charger_occupation']['in_charge_fast']
    files_to_track_performance=make_files_to_track_performance(files_to_track_performance,'update',performance_update)
    apply_actions(files_to_track_performance,fleet_attribute)
    path=path_dic['auxilary']
    open(path+'time_in_simulation.txt', 'w').close()
    print('charging script: time ',time_in_simulation)
    with open(path+'time_in_simulation.txt', "w") as f:
        f.write(str(time_in_simulation))


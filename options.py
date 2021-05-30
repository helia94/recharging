import pandas as pd
import json
import numpy as np
import glob
from pandas.io.json import json_normalize
from geopandas import read_file
from  ast import literal_eval
import os
import datetime
from requests import post
from shapely.geometry import Point
from charging_prams import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class XServer:
    """class defines Requests for 1:1 routing and Isochrone routing
    """ 
    def make_route_request(self, profile, waypoints):

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
                                "routingType": "CONVENTIONAL",#"HIGH_PERFORMANCE_ROUTING",
                            
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
        
#         request["userLogs"] += {"1-1 Routing Request"}
        
        return request   

    def send_route_request(self, request):
        
        
        # returns a string representing a json object
        json_data = json.dumps(request)

        UrlCalculateRoute = "http://localhost:50000" + "/services/rs/XRoute/experimental/calculateRoute"
        header = {"content-type": "application/json;charset=utf-8"}
        json_resp = post(url=UrlCalculateRoute, data=json_data, headers=header)
#         print(json.loads(json_resp.text))

        if json_resp.status_code == 200:
            response = json_resp.text
            pyres = json.loads(response)

        else:
            print('xs2 Request failed 1')
            print(json.loads(json_resp.text))
            return -99,-99,-99
        distance = 0
        if "$type" in pyres:
      
            distance = pyres.get("distance")
            travel_time = pyres.get("travelTime")
            report = pyres.get("report")

            way_points = pyres.get("waypoints")

        else:
            "Couldn't read data from xs"
        return distance, travel_time, way_points#, report
                                


def link_consumption (distance, driving_range=150):
    consumption=100*(distance/(driving_range*1000))
    return consumption


list_of_bands=[5, 10, 15, 20]

def read_options():  
    columns_used=[#'ExpectedDropoffTime.Nanos', 
                  'ExpectedDropoffTime.Seconds',
       #'ExpectedPickupTime.Nanos', 
       #'ExpectedPickupTime.Seconds',
       'TourTripInsertion.DropoffStopIndex',
       'TourTripInsertion.PickupStopIndex',
#        'TourTripInsertion.TravelFromDropoff.Nanos',
#        'TourTripInsertion.TravelFromDropoff.Seconds',
#        'TourTripInsertion.TravelFromPickup.Nanos',
#        'TourTripInsertion.TravelFromPickup.Seconds',
#        'TourTripInsertion.TravelToDropoff.Nanos',
#        'TourTripInsertion.TravelToDropoff.Seconds',
#        'TourTripInsertion.TravelToPickup.Nanos',
#        'TourTripInsertion.TravelToPickup.Seconds', 
        'TripId', 
        'VehicleId'] 
    path=path_dic['from_simulator']
    with open(path+'optionsOut.json') as f:
        json_data = json.load(f)
    if len(json_data)==0:
        return None, None, 'fail'
    df_options=json_normalize(json_data)
    del json_data
    df_options=df_options[columns_used]
    trip_id=df_options.loc[0,'TripId']
    if STAGE==3:      
        #read trip_id
        path=path_dic['from_simulator']
        with open(path+'TripRequests.json','r') as f:
            json_data = json.load(f)
        trip_from_to=[[[trip['FromX'],trip['FromY']],[trip['ToX'],trip['ToY']]] for trip in json_data if trip['Id']==str(trip_id)][0]
        del json_data
    
  
        #read vehicle waypoints
        path=path_dic['auxilary']
        df_vehicle_tours_and_SoC=pd.read_csv(path+'df_vehicle_tours_and_SoC.csv',sep=';')
        df_options=pd.merge(df_options,df_vehicle_tours_and_SoC,how='left',on='VehicleId')
        df_options.rename(columns={'waypoints':'waypoints_old'},inplace=True)
        
        #read nearst_charger data
        
        path=path_dic['scenario']
        gdf_ReacableArea_ = read_file( path+"gdf_ReacableArea.SHP")    
        
        #update and get insertion cost
        def update_waypoints(list_waypoints,index_from,index_to):
    #         print((list_waypoints,index_from,index_to+1,trip_from_to))
    #         print(list_waypoints)
            list_waypoints_new=list_waypoints[:]
            list_waypoints_new.insert(index_from+1,trip_from_to[0])
    #         print(list_waypoints)
            list_waypoints_new.insert(index_to+2,trip_from_to[1])
    #         print(list_waypoints)
            return list_waypoints_new
    #     print(df_options['waypoints_old'].loc[0])
        df_options['waypoints_old']=df_options['waypoints_old'].apply(literal_eval)
    #     print(df_options['waypoints_old'].loc[0])
        df_options['waypoints_new']=df_options.apply(lambda row: update_waypoints(row['waypoints_old'],row['TourTripInsertion.PickupStopIndex'],row['TourTripInsertion.DropoffStopIndex']),axis=1)
        x_server = XServer()
        

        def get_cunsumption_TT(waypoints, x_server, link_consumption):
            if len(waypoints)<=1:
                return 0
            profile="car"
##            print('waypoints',waypoints)
            route_request=x_server.make_route_request( profile, waypoints)
            distance, travel_time, _=x_server.send_route_request(route_request)
            cunsumption=link_consumption(distance)
            return cunsumption
        def get_to_charger_consumption(coord):
            for band in list_of_bands:
    #             reach=
##                print('coord',coord)
                point=Point(coord)    
                if gdf_ReacableArea_['geometry'].contains(point).sum()>0:
                    consumption=link_consumption(band*1000)
                    return consumption
                return 50

        df_options['consumption_old']=df_options.apply(lambda row:
                                                    get_cunsumption_TT(row['waypoints_old'],x_server,link_consumption), axis=1)
        df_options['consumption_new']=df_options.apply(lambda row:
                                                    get_cunsumption_TT(row['waypoints_new'],x_server,link_consumption), axis=1)
    #     print(df_options[['consumption_old','consumption_new']])
        df_options['insertion_cost']=df_options['consumption_new']-df_options['consumption_old']
        



        for type_ in ['old','new']:
            df_options['last_drop_off_'+type_]=df_options['waypoints_'+type_].apply(lambda x: x[-1])
            
            df_options['to_charger_consumption_'+type_]=df_options['last_drop_off_'+type_].apply(lambda x: get_to_charger_consumption(x))
        df_options['charge_buffer']=df_options['SoC_now']-df_options['consumption_new']-df_options['to_charger_consumption_new']
        df_options['has_enough_battery']=df_options['charge_buffer'].apply(lambda x:x>5)
##        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
##            print(df_options[['VehicleId','SoC_now','consumption_new','to_charger_consumption_new','insertion_cost','ExpectedDropoffTime.Seconds']])
##        if df_options.loc[0]['has_enough_battery']==0:
##            print('Mismatch should happen: first choice',str(df_options.loc[0]['VehicleId']),'does not have enough charge')
        df_options=df_options[df_options['has_enough_battery']==1]
        
        df_options['must_charge']=(df_options['SoC_now']-df_options['consumption_new']).apply(lambda x:x<LB_SoC_to_work)

        df_options['metric_for_SoC']=1
        if SoC_in_assignment and STAGE==3:
            path=path_dic['auxilary']
            with open(path+'soc_metric_t.json', "r") as f:
                soc_metric_t=json.load(f)
            def get_metric(soc):
                metric_list=[]
                for above in list(soc_metric_t.keys()):
                    if soc>=int(above):
                        metric_list+=[soc_metric_t[above]]
                if len(metric_list)>0:
                    metric=max(metric_list)
                else:
                    metric=1
                return metric 

            df_options['metric_for_SoC']=df_options['SoC_now'].apply(get_metric)
        
        df_options['time_in_simulation']=time_in_simulation
        if len(df_options)==0:
            print('no option had charge')
            rejected=pd.DataFrame(data={'time':[time_in_simulation],'TripId':[trip_id]})
            path=path_dic['results']+'rejected_by_lack_of_charge'+'.csv'
            if os.path.exists(path):
                with open(path, 'a') as f:
                    rejected.to_csv(f, header=False, index=False,sep=';')
            else:
                rejected.to_csv(path, index=False,sep=';')
            return None,None, 'fail'
        return df_options,df_vehicle_tours_and_SoC, 'succes'
    else:
        return df_options, None, 'succes'

def get_utilities(df_options):
    df_options['utility']=R_weights['insertion']*df_options['insertion_cost']+R_weights['cost_to_charger']*df_options['must_charge']*(
        df_options['to_charger_consumption_new']-df_options['to_charger_consumption_old'])+R_weights['metric']*df_options['metric_for_SoC']
#     -1*df_options['ExpectedDropoffTime.Seconds']
    return df_options
        

path=path_dic['auxilary']
with open(path+'time_in_simulation.txt', "r") as f:
    time_in_simulation=int(f.read())
if time_in_simulation==0:
    STAGE=2
df_options,df_vehicle_tours_and_SoC, code =read_options()
if code=='succes':
    if SoC_in_assignment and STAGE==3:
        df_options=get_utilities(df_options)
        df_options.index=np.arange(len(df_options))
        max_id=df_options['utility'].argmin()
##        if max_id!=0:
##            print('choices are different')
        opt_vehicle=df_options.loc[df_options.index==max_id,'VehicleId'].item()
    else:
        max_id=0 # to get original best option
        df_options.index=np.arange(len(df_options))
        opt_vehicle=df_options.loc[df_options.index==max_id,'VehicleId'].item()
    request={"TripId":df_options.loc[df_options.index==max_id,'TripId'].item(),"VehicleId":opt_vehicle}
##    print(request)
    path=path_dic['from_simulator']
    with open(path+'BindingVehicleChoice.json', "w") as f:
        json.dump(request, f)
    if  STAGE==3:
##        print(df_vehicle_tours_and_SoC.at[df_vehicle_tours_and_SoC.index[df_vehicle_tours_and_SoC['VehicleId']==opt_vehicle].item(),'waypoints'])
##        print([df_options.loc[max_id,'waypoints_new'].item()])
##        df_vehicle_tours_and_SoC.at[df_vehicle_tours_and_SoC.index[df_vehicle_tours_and_SoC['VehicleId']==opt_vehicle].item(),'waypoints']=df_options.loc[max_id,'waypoints_new']
##        print(df_vehicle_tours_and_SoC.at[df_vehicle_tours_and_SoC.index[df_vehicle_tours_and_SoC['VehicleId']==opt_vehicle].item(),'waypoints'])
        path=path_dic['auxilary']
        df_vehicle_tours_and_SoC.to_csv(path+'df_vehicle_tours_and_SoC.csv',sep=';',index=False)
        if SoC_in_assignment:
##            print('max_id',max_id)
            chosen_vehicles=df_options.loc[df_options.index==max_id,:]            
            ptv_chosen_vehicles=df_options.loc[df_options.index==0,:]
            for name in ['chosen_vehicles','ptv_chosen_vehicles']:
                path=path_dic['results']+name+'.csv'
                if os.path.exists(path):
                    with open(path, 'a') as f:
                        eval(name).to_csv(f, header=False, index=False,sep=';')
                else:
                    eval(name).to_csv(path, index=False,sep=';')
else:
    path=path_dic['from_simulator']
    with open(path+'BindingVehicleChoice.json', "w") as f:
        json.dump

##path=path_dic['from_simulator']
##with open(path+'optionsOut.json', "w") as f:
##    json.dump
#update tours in file 
# print(df_options.loc[max_id,'waypoints_new'])

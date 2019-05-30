
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import enum
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Tuple
import pickle
import pandas as pd
from random import randint, getrandbits
import datetime
from datetime import timedelta
import os
from datetime import date
import copy
from timeit import default_timer as timer

from vrp import *

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions*2)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        return self.layers(x)

class INFORMSsolver:
    def __init__(self, expdir='../../Exp', year='2015', route_type=1, vehicle_type=1, restore_snapshot=None):


        # route_type
        # 1: static 
        # 2: dynamic

        # vehicle_type
        # 1: non-AV
        # 2: AV


        current_date = pd.to_datetime(f"{year}-01-01")

        
        
        datadir = f'{expdir}/{year}'
        commondir = f'{expdir}'

        distance_matrix = pickle.load(open(f'{commondir}/dist_mat_9361.dump', 'rb'))
        location_index_dic = pickle.load(open(f'{commondir}/location_index_9361.dump', 'rb'))
        
        location = pd.read_excel('../../Informs_Data_Set/Input_Cost%2C+Location.xlsx')


        shipment_master_df = pickle.load(open(f'{datadir}/shipment.dump', 'rb'))
        vdc_info_dic = pickle.load(open(f'{datadir}/vdc_info.dump', 'rb'))

        if route_type == 1:
            route_dic = pickle.load(open(f'{datadir}/route_static.dump', 'rb'))
        else:
            route_dic = pickle.load(open(f'{datadir}/route_dynamic.dump', 'rb'))


        VDC = vdc_info_dic.keys()



        Links = [(i,j) for i in VDC for j in VDC if i != j]

        Plants = ('3A','FF','RO','SO')


        # maximum number of days a car can stay at a VDC
        MAX_STAY_DAYS = 7

        # min dwell time (days)
        MIN_DWELL_TIME = 2

        # decision interval (hours)
        DECISION_INTERVAL = 6

        # time spent at a VDC before delivery (days)
        DELIVERY_TIME = 2

        # how many future time periods the state considers
        NUM_LOOKING_FORWARD = 8


        # General information of each link
        # Link_info['3A']['SO'] = {'distance':..., 'time':..., 'rail':...}
        Link_info = {
            i: {
                j: {
                    'distance': distance_matrix[location_index_dic[i]][location_index_dic[j]],
                    'rail': vdc_info_dic[i]['rail'] and vdc_info_dic[j]['rail']
                }
                for j in VDC
                if i != j
            }
            for i in VDC
        }

        for i in VDC:
            for j in VDC:
                if i != j:
                    Link_info[i][j]['time'] = Link_info[i][j]['distance'] / (10.0 if Link_info[i][j]['rail'] else 30.0)

        # cars at each vdc
        VDC_all_df = {
            v: pd.DataFrame(columns=pd.Index(['vin', 'vdc', 'arrival_time', 'dealer', 'plant', 'pd_time', 'route', 'arcs', 'next', 'delivery'], dtype='object'))
            for v in VDC
        }
        
        
        # VRP solver
        self.vrp = VRP(location, distance_matrix, location_index_dic, list(VDC))
        

        model = DQN(NUM_LOOKING_FORWARD * 2, 1)
        model.load_state_dict(torch.load(f'{commondir}/INTERVER6_model_static_dic_256_10000_100.pt'))
        model.eval()
        
        self.model = model

        # problem data
        self.year = year
        self.current_date = current_date
        self.expdir = expdir
        self.datadir = datadir
        self.commondir = commondir
        self.distance_matrix = distance_matrix
        self.location_index_dic = location_index_dic
        self.shipment_master_df = shipment_master_df
        self.vdc_info_dic = vdc_info_dic
        self.route_dic = route_dic
        self.route_type = route_type
        self.vehicle_type = vehicle_type
        self.VDC = VDC
        self.Links = Links
        self.Plants = Plants
        self.Link_info = Link_info
        self.VDC_all_df = VDC_all_df


        # parameters
        self.MAX_STAY_DAYS = MAX_STAY_DAYS
        self.MIN_DWELL_TIME = MIN_DWELL_TIME
        self.DECISION_INTERVAL = DECISION_INTERVAL
        self.DELIVERY_TIME = DELIVERY_TIME
        self.NUM_LOOKING_FORWARD = NUM_LOOKING_FORWARD
        

        
        
        # results
        self.result_vdc_to_vdc_df = pd.DataFrame(columns=pd.Index(['depart_vdc', 'depart_time', 'arrival_vdc', 'arrival_time', 'method', 'vins', 'num_vins'], dtype='object'))
        self.result_delivery_df = pd.DataFrame(columns=pd.Index(['vdc', 'shipment_id', 'path', 'vins', 'cost', 'method'], dtype='object'))
        self.result_route_details_df = pd.DataFrame(columns=pd.Index(['vin', 'loc', 'arrive_time', 'depart_time', 'depart_mode', 'shipment_id', 'shipment_cost'], dtype='object'))


        # restore status from snapshot
        if restore_snapshot is not None:
            snapshot = pickle.load(open(restore_snapshot, 'rb'))
            self.current_date = snapshot['date']
            self.VDC_all_df = {**self.VDC_all_df, **snapshot['vdc_status']}
            self.result_vdc_to_vdc_df = snapshot['vdc_to_vdc']
            self.result_delivery_df = snapshot['delivery']
            

        

    def append_cars_arrive_today(self):
        for p in self.Plants:
            arrive_today_df = self.shipment_master_df[
                (self.shipment_master_df.Plant.values == p) &
                (self.shipment_master_df.Plant_Arrival_Time >= self.current_date) &
                (self.shipment_master_df.Plant_Arrival_Time < self.current_date + timedelta(days=1))
            ]
            for index, row in arrive_today_df.iterrows():
                routes = self.route_dic[(row.Plant, row.Dealer)]['path']
                self.VDC_all_df[p].loc[row.VIN] = [
                    row.VIN, 
                    row.Plant, 
                    row.Plant_Arrival_Time, 
                    str(row.Dealer), 
                    row.Plant, 
                    row.Plant_Arrival_Time, 
                    routes, 
                    set([a for route in routes for a in route]),
                    None, 
                    row.Plant not in self.VDC
                ]

                
    def determine_next_dest(self):
        if self.route_type == 1: # static routing

            def next_dest_in_route(r,v):
                for a in r:
                    if a[0] == v:
                        return a[1]
                else:
                    return None

            for vdc in self.VDC:
                next_dest = [
                    str(next_dest_in_route(row.route[0], vdc))
                    for vin, row in self.VDC_all_df[vdc].iterrows()
                ]

                self.VDC_all_df[vdc].next = next_dest
                
        elif self.route_type == 2: # dynamic routing
            
            for v in self.VDC:
                self.determine_dynamic_routing(v)
            
        
        # determine delivery
        for v in self.VDC:
            self.VDC_all_df[v].delivery = ~self.VDC_all_df[v].next.isin(self.VDC)
            
            
    def determine_dynamic_routing(self, vdc):
        
        
        # def dertermine_dynamic_routing(self, vdc):
        cars_to_determine_df = self.VDC_all_df[vdc]

        NC_dic = {
            vin: [a[1] for a in arcs if a[0]==vdc]
            for (vin,arcs) in zip(cars_to_determine_df.index, cars_to_determine_df.arcs.values)
        }

        # reset next
        cars_to_determine_df.loc[~cars_to_determine_df.delivery != True].next = None

        # True if the car is to deliver 
        # We assume that the car is to deliver if there is a dealer node in the next candidate set
        cars_to_deliver = [
            any(map(lambda i: i not in self.VDC, next_nodes))
            for vin, next_nodes in NC_dic.items()
        ]

        # set delivery
        cars_to_determine_df.delivery = cars_to_deliver
        # set next as dealer if delivery car
        cars_to_determine_df.loc[cars_to_determine_df.delivery, 'next'] = cars_to_determine_df.loc[cars_to_determine_df.delivery, 'dealer']

        # no delivery cars
        cars_to_determine_df = cars_to_determine_df[~cars_to_determine_df.delivery]


        # determine next destination of all cars in the vdc
        while sum(self.VDC_all_df[vdc].next.isnull()) > 0:

            NC_df = pd.DataFrame.from_dict(
            {
                vin: [a[1] for a in arcs if a[0]==vdc]
#                 for (vin,arcs) in zip(cars_to_determine_df.index, cars_to_determine_df.arcs.values)
                for (vin,arcs) in zip(self.VDC_all_df[vdc][self.VDC_all_df[vdc].next.isnull()].index, self.VDC_all_df[vdc][self.VDC_all_df[vdc].next.isnull()].arcs.values)
            },
            orient='index'
            )

            # find the most available next destination
            most_freq_next_node = pd.concat([NC_df[col].value_counts()*(len(NC_df.columns)-col) for col in NC_df.columns]).groupby(level=0).sum().sort_values(ascending=False).iloc[:1].index[0]

            # cars that can be sent to the most_freq_next_node
            vins_to_most_freq_next_node = NC_df[0] == most_freq_next_node
            if len(NC_df.columns) > 1:
                for col in NC_df.columns[1:]:
                    vins_to_most_freq_next_node |= NC_df[col] == most_freq_next_node


            self.VDC_all_df[vdc].loc[vins_to_most_freq_next_node.index, 'next'] = most_freq_next_node
        

    def get_state(self, link, period):
        
        return g_get_state(self.current_date, period, self.DECISION_INTERVAL, self.MIN_DWELL_TIME, self.NUM_LOOKING_FORWARD, link)

    
    def get_action(self, state):
        state = torch.autograd.Variable(torch.FloatTensor(state).unsqueeze(0))
        q_value = self.model(state)
        action = 1 if q_value[0,0]<q_value[0,1] else 0
#         # dummy model...
#         return state[0] >= 7

        return action == 1

    def send_cars(self, link, period, num_to_send):

        current_time = (self.current_date) + timedelta(hours=period*self.DECISION_INTERVAL)

        from_vdc, to_vdc = link[0], link[1]

        from_vdc_df = self.VDC_all_df[from_vdc]
        # all cars between two vdcs
        link_all_df = from_vdc_df[(from_vdc_df.next.values == to_vdc)]

        cars_to_send_df = link_all_df[link_all_df.arrival_time <= current_time - timedelta(days=self.MIN_DWELL_TIME)]

        # send cars that stayed longest
        cars_to_send_df = cars_to_send_df.sort_values('arrival_time', axis=0)[:num_to_send]
        
        # data for route details
        vins = list(cars_to_send_df.index)
        loc = [from_vdc] * len(cars_to_send_df)
        prev_arrive_times = list(cars_to_send_df.arrival_time)

        # remove cars to send
        self.VDC_all_df[from_vdc] = from_vdc_df.drop(cars_to_send_df.index)

        # we can send the cars after the time at which the latest car arrived
        time_to_send = cars_to_send_df.arrival_time.max() + timedelta(days=self.MIN_DWELL_TIME)
        
#         print(time_to_send)
#         print(self.current_date)
        
        if time_to_send < self.current_date:
            time_to_send = self.current_date

        link_info = self.Link_info[from_vdc][to_vdc]

        arrive_time = time_to_send + timedelta(hours=link_info['time'])

        depart_time = [time_to_send] * len(cars_to_send_df)
        depart_mode = ['R' if link_info['rail'] else 'T'] * len(cars_to_send_df)
        shipment_cost = [self.calc_transport_cost(link, len(cars_to_send_df))] * len(cars_to_send_df)

        # copy detaframe to manupulation
        cars_to_arrive_df = cars_to_send_df.copy()

        # reset next destinations of cars to send
        cars_to_arrive_df.next = None

        cars_to_arrive_df.vdc = to_vdc
        cars_to_arrive_df.arrival_time = arrive_time
        cars_to_arrive_df.delivery = False

        #                 display(cars_to_arrive_df)

        # append cars to arrive at the destination
        self.VDC_all_df[to_vdc] = pd.concat([self.VDC_all_df[to_vdc], cars_to_arrive_df])   
        
        # update result dataframe
        self.result_vdc_to_vdc_df = self.result_vdc_to_vdc_df.append({
            'depart_vdc': from_vdc, 
            'depart_time': time_to_send, 
            'arrival_vdc': to_vdc,
            'arrival_time': arrive_time, 
            'method': 'rail' if link_info['rail'] else 'truck',
            'vins': list(cars_to_arrive_df.index), # index represents vin...
            'num_vins': len(cars_to_arrive_df)
        }, ignore_index=True)
        
        self.result_route_details_df = self.result_route_details_df.append([{
            'vin': v,
            'loc': l,
            'arrive_time': p,
            'depart_time': dt,
            'depart_mode': dm,
            'shipment_cost': s
        } for v,l,p,dt,dm,s in zip(vins, loc, prev_arrive_times, depart_time, depart_mode, shipment_cost)], ignore_index=True)
        
        
    def calc_transport_cost(self, link, num_cars):
        from_vdc, to_vdc = link[0], link[1]
        link_info = self.Link_info[from_vdc][to_vdc]
        if link_info['rail']:
            return ((2000 + link_info['distance']*3) * (((num_cars-0.1) // 20)+1)) / num_cars # rail
        else:
            return ((200 + link_info['distance']*4) * (((num_cars-0.1) // 10)+1)) / num_cars # truck
        
        
    def process_full_load(self, state, link, period):
        from_vdc, to_vdc = link[0], link[1]
        
        link_info = self.Link_info[from_vdc][to_vdc]

        full_load = 20 if link_info['rail'] else 10

        if state[0] >= full_load:
            num_full_loads = state[0] // full_load
            self.send_cars((from_vdc, to_vdc), period, num_full_loads*full_load)
            return self.get_state(link, period)
        else:
            return state
        
    def process_today_vdc(self, dview=None):
        
        if dview is None:

            global g_VDC_all_df
            g_VDC_all_df = self.VDC_all_df

            for period in range(int(24 / self.DECISION_INTERVAL)):
                for link in self.Links:
                    from_vdc, to_vdc = link[0], link[1]

                    state = self.get_state(link, period)

                    # process full loads
                    state = self.process_full_load(state, link, period)

                    action = self.get_action(state)

                    if action == 1: # send
                        self.send_cars(link, period, state[0])
                    else:
                        self.send_cars_stayed_too_long(link)
        else:
            if dview is not None:
                
                for period in range(int(24 / self.DECISION_INTERVAL)):
                    link_list = self.Links
                    period_list = [period] * len(link_list)
                    current_date_list = [self.current_date] * len(link_list)
                    decision_interval_list = [self.DECISION_INTERVAL] * len(link_list)
                    min_dwelltime_list = [self.MIN_DWELL_TIME] * len(link_list)
                    num_looking_forward_list = [self.NUM_LOOKING_FORWARD] * len(link_list)

                    dview['g_VDC_all_df'] =  self.VDC_all_df
                    
                    state_list = dview.map_sync(g_get_state, current_date_list, period_list, decision_interval_list, min_dwelltime_list, num_looking_forward_list, link_list)

                    for state, link in zip(state_list, link_list):
                        # process full loads
                        state = self.process_full_load(state, link, period)

                        action = self.get_action(state)

                        if action == 1: # send
                            self.send_cars(link, period, state[0])
                        else:
                            self.send_cars_stayed_too_long(link)

        
    
    def send_cars_stayed_too_long(self, link):
        from_vdc, to_vdc = link[0], link[1]
        from_vdc_df = self.VDC_all_df[from_vdc]
        if len(from_vdc_df) > 0:
            # all cars between two vdcs
            link_all_df = from_vdc_df[(from_vdc_df.next.values == to_vdc)]

            if len(link_all_df) > 0:
                cars_to_send_df = link_all_df[link_all_df.arrival_time <= self.current_date - timedelta(days=self.MAX_STAY_DAYS)]

                if len(cars_to_send_df) > 0:
                    self.send_cars(link, 0, 20)
                
                

    def update_inventory_status(self):
        inv_dic = {
            v : [
                sum((self.VDC_all_df[v].delivery==False) & (self.VDC_all_df[v].arrival_time<=self.current_date)), 
                sum((self.VDC_all_df[v].delivery==True) & (self.VDC_all_df[v].arrival_time<=self.current_date)), 
                sum(self.VDC_all_df[v].arrival_time<=self.current_date)
            ]
            for v in self.VDC
        }

        inv_df = pd.DataFrame.from_dict(inv_dic, orient='index', columns=['To VDC', 'To dealer', 'Total']).transpose()
        self.inventory_df = inv_df 
        return inv_df
        
    def solve_delivery(self, vdc, drop_delivered_cars=True):
        cars_to_deliver_df = self.VDC_all_df[vdc][self.VDC_all_df[vdc].delivery == True]
        # cars_to_deliver_df.index.name = 'vin'

        vrp_cost, _, adjusted_vrp_sol = self.vrp.solve_vrp(vdc, str(self.current_date)[:10], cars_to_deliver_df, 3, log=False)


        delivered_cars = [
            vin
            for shipment in adjusted_vrp_sol
            for vin,_,_ in adjusted_vrp_sol[shipment]['vins']
        ]


        # update result
        self.update_result_for_delivery(vdc, adjusted_vrp_sol)

        if drop_delivered_cars:
            # drop delivered cars
            self.VDC_all_df[vdc] = self.VDC_all_df[vdc][~self.VDC_all_df[vdc].vin.isin(delivered_cars)]



        return delivered_cars
    
        
    
    def solve_delivery_parallel(self, dview, drop_delivered_cars=True):    
        all_cars_to_deliver_df = [
            swlf.VDC_all_df[vdc][self.VDC_all_df[vdc].delivery == True]
            for vdc in self.VDC
        ]

        all_vrp_solutions_list = dview.map_sync(par_solve_vrp, self.VDC, [str(self.current_date)[:10]] * len(self.VDC), all_cars_to_deliver_df, [3] * len(self.VDC), [False] * len(self.VDC))

        all_delivered_cars = {
            vdc: [
                vin
                for shipment in adjusted_vrp_sol
                for vin,_,_ in adjusted_vrp_sol[shipment]['vins']
            ]   
            for (vdc, (vrp_cost, _, adjusted_vrp_sol)) in zip(self.VDC, all_vrp_solutions_list)
        }


        # update result
        for (vdc, (vrp_cost, _, adjusted_vrp_sol)) in zip(self.VDC, all_vrp_solutions_list):
            self.update_result_for_delivery(vdc, adjusted_vrp_sol)

        if drop_delivered_cars:
            # drop delivered cars
            for vdc in self.VDC:
                self.VDC_all_df[vdc] = self.VDC_all_df[vdc][~self.VDC_all_df[vdc].vin.isin(all_delivered_cars[vdc])]

        return {v:len(all_delivered_cars[v]) for v in self.VDC}

    def update_result_for_delivery(self, vdc, adjusted_vrp_sol):
        if len(adjusted_vrp_sol) > 0:
            self.result_delivery_df = self.result_delivery_df.append([{
                'vdc': vdc,
                'shipment_id': shipment,
                'path': sol['path'],
                'vins': sol['vins'],
                'cost': sol['cost'],
                'method': 'truck',
            } for shipment, sol in adjusted_vrp_sol.items()], ignore_index=True)
        
        for shipment, sol in adjusted_vrp_sol.items():
            
            vins = [v[0] for v in sol['vins']]
            loc = [vdc] * len(vins)
            prev_arrive_times = list(self.VDC_all_df[vdc].loc[vins].arrival_time)
            depart_time = [sol['path'][0][1]] * len(vins)
            depart_mode = ['T'] * len(vins)
            shipment_id = [shipment] * len(vins)

            self.result_route_details_df = self.result_route_details_df.append([{
                'vin': v,
                'loc': l,
                'arrive_time': p,
                'depart_time': dt,
                'depart_mode': dm,
                'shipment_id': s
            } for v,l,p,dt,dm,s in zip(vins, loc, prev_arrive_times, depart_time, depart_mode, shipment_id)], ignore_index=True)

            vins = [v[0] for v in sol['vins']]
            loc = list(self.VDC_all_df[vdc].loc[vins].dealer)
            arrive_times = [v[1] for v in sol['vins']]
            depart_time = arrive_times
            depart_mode = ['T'] * len(vins)

            self.result_route_details_df = self.result_route_details_df.append([{
                'vin': v,
                'loc': l,
                'arrive_time': p,
                'depart_time': dt,
                'depart_mode': dm
            } for v,l,p,dt,dm in zip(vins, loc, arrive_times, depart_time, depart_mode)], ignore_index=True)
    
    
    def save_today_snapshot(self, savedir='res'):
        today_snapshot = {
            'date': self.current_date,
            'vdc_status': self.VDC_all_df,
            'vdc_to_vdc': self.result_vdc_to_vdc_df,
            'delivery': self.result_delivery_df,
            'route_details_today': self.result_route_details_df[(self.result_route_details_df.depart_time >= self.current_date-timedelta(days=1)) & (self.result_route_details_df.depart_time < self.current_date)]
        }

        pickle.dump(today_snapshot, open(f'{savedir}/{self.current_date.strftime("%Y-%m-%d")}.dump', 'wb'))


    def solve(self, from_date=None, to_date=None, dview=None, save_dir='res'):
        if from_date is not None:
            self.current_date = pd.to_datetime(from_date)

        if to_date is not None:
            end_date = pd.to_datetime(to_date)
        else:
            end_date = pd.to_datetime(f'{self.year}-12-31')

        start_solve = timer()

        save_everyday_status = True

        while self.current_date <= end_date:


            print(f'{self.current_date.strftime("%Y/%m/%d")} Solving time: {timer()  - start_solve:0.2f} seconds')

            self.append_cars_arrive_today()

            self.determine_next_dest()

            print(f'\tProcess VDC-VDC decisions{(" (parallel:use "+str(len(dview))+" cores)") if dview is not None else ""}', end='')
            start_vdc = timer()
            self.process_today_vdc(dview)
            print(f': {timer()  - start_vdc:0.2f} seconds.')

            self.determine_next_dest()

            start_vrp = timer()
            if dview is not None:
                print(f'\tSolve delivery problems (parallel:use {len(dview)} cores): ', end='')
                all_deliverted_cars = self.solve_delivery_parallel(dview)
            else:
                print(f'\tSolve delivery problems: ', end='')
                all_deliverted_cars ={}
                for v in self.VDC:
                    print(f'{v}', end=' ')
                    delivered_cars = self.solve_delivery(v)
                    all_deliverted_cars[v] = len(delivered_cars)
            print(f': {timer()  - start_vrp:0.2f} seconds.')

            self.current_date += timedelta(days=1)

            self.update_inventory_status()
            self.inventory_df.loc['Delivered today'] = (all_deliverted_cars)

            print('\tInventory status:')
            display(self.inventory_df)


            if save_everyday_status:
                self.save_today_snapshot(savedir=save_dir)


# helper functions for parallel
def par_solve_vrp(vdc, date, cars_to_deliver_df, period, log=True):
    return vrp.solve_vrp(vdc, date, cars_to_deliver_df, period, log)

def g_get_state(current_date, period, decision_interval, min_dwell_time, num_looking_forward, link):

    current_time = (current_date) + timedelta(hours=period*decision_interval)

    from_vdc, to_vdc = link[0], link[1]

    state = []
                  
#     global g_VDC_all_df
                  
    from_vdc_df = g_VDC_all_df[from_vdc]
    to_vdc_df = g_VDC_all_df[to_vdc]

    if len(from_vdc_df) > 0:

        # all cars between two vdcs
        link_all_df = from_vdc_df[(from_vdc_df.next.values == to_vdc)]

        if len(link_all_df) > 0:
            # numbers of cars that can be transported at t, t+1, ..., t+NUM_LOOKING_FORWARD
            link_num_cars = [
                sum(link_all_df.arrival_time <= current_time - timedelta(days=min_dwell_time) + timedelta(hours=t*decision_interval))
                for t in range(num_looking_forward)
            ]
        else:
            link_num_cars = [0] * (num_looking_forward)

        state.extend(link_num_cars)

        if len(to_vdc_df) > 0:
            # numbers of cars of destination vdc at t, t+1, ..., t+NUM_LOOKING_FORWARD
            inv_cars = [
                sum(to_vdc_df.arrival_time <= current_time - timedelta(days=min_dwell_time) + timedelta(hours=t*decision_interval))
                for t in range(num_looking_forward)
            ]
        else:
            inv_cars = [0] * (num_looking_forward)

        state.extend(inv_cars)

    else:
        state = [0] * (num_looking_forward) * 2

    # construct a state
    state = np.array(state)

    return state

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

pd.options.display.max_columns = None

# import math, random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.autograd as autograd

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# import networkx as nx


import time

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class VRP:
#     def __init__(self, location, dist, ind, vdc_list):
    def __init__(self, vdc_list):
        # self.location = g_location
        # self.dist = g_dist
        # self.ind = g_ind
        self.vdc_list = vdc_list


    def create_data_model(self, day, dff4, vdc, next_day, more_ten_vin, check_dealer, check_vin, sol_check_dealer,
                          delta, ten_solution_delta, ten_sol_route_dic,check_dealer_del):

        Arrive = {}
        Arrival_time = {}
        p_time = {}
        p_vin_time = {}
        dealer = []
        vin = []
        VDC = vdc

        ind = self.ind
        dist = self.dist

        dff4 = dff4[dff4.vdc.values == vdc]

        Arrive = dff4.groupby('dealer')['vin'].apply(list).to_dict()
        Arrival_time = dff4.groupby('dealer')['arrival_time'].apply(list).to_dict()
        p_time = dff4.groupby('dealer')['pd_time'].apply(list).to_dict()

        p_vin_time = dff4.groupby('vin')['pd_time'].apply(list).to_dict()
        arrive_vin_time = dff4.groupby('vin')['arrival_time'].apply(list).to_dict()  # ?

        dealer = list(dff4.dealer)
        vin = list(dff4.vin)

        #     print('Arrival time',Arrival_time)
        #     print('Arrive',Arrive)
        #     print('dealer',dealer)
        #     print('p_time',p_time)
        #     print('p_vin_time',p_vin_time)
        #     print('vin',vin)

        # print('dealer',dealer)

        t_dealer = list(set(dealer))
        # print('check_dealer', check_dealer)
        if len(sol_check_dealer[vdc]) > 0:
            set_t_dealer = set(t_dealer)
            set_sol_check_dealer = set(sol_check_dealer[vdc])
            set_dealer = set_t_dealer - set_sol_check_dealer
            t_dealer = list(set_dealer)

        t_ind_all = [VDC] + t_dealer
        # print('t_ind_all', t_ind_all)

        t_ind_num = []
        for i in t_ind_all:
            t_ind_num.append(ind[i])
        # print('t_ind_num', t_ind_num)

        first_len = len(t_ind_num)
        # print('first_len', first_len)

        t_ind_dic = {}
        for i in range(len(t_ind_num)):
            t_ind_dic[i] = t_ind_num[i]
        # print('t_ind_dic', t_ind_dic)

        t_ind_name = {}
        for i in range(len(t_ind_num)):
            t_ind_name[i] = t_ind_all[i]
        # print('t_ind_name', t_ind_name)

        t_ind_vin_dic = {}
        t_ind_dealer_dic = {}
        for i in range(1, len(t_ind_name)):
            t_ind_vin_dic[list(t_ind_name.keys())[list(t_ind_name.values()).index(t_ind_name[i])]] = Arrive[
                t_ind_name[i]]
            t_ind_dealer_dic[list(t_ind_name.keys())[list(t_ind_name.values()).index(t_ind_name[i])]] = t_ind_name[i]
        #     print('t_ind_vin_dic',t_ind_vin_dic)
        #     print('t_ind_dealer_dic',t_ind_dealer_dic)

        t_demand = {}

        test_demand = {}
        for i in t_ind_all:
            str_date = str(day)
            Y = int(str_date[0:4])
            M = int(str_date[5:7])
            D = int(str_date[8:10])
            ship_day = date(Y, M, D)
            
            # print('####',ship_day)
            # ship_day.strftime("%Y/%m/%d")

            if i == vdc:
                t_demand[ind[i]] = [0]
                test_demand[ind[i]] = [0]
            else:
                if ind[i] not in t_demand:
                    test_demand[ind[i]] = [len(Arrive[i])]
                    if len(Arrive[i]) > 10:
                        t_demand[ind[i]] = [len(Arrive[i]) % 10]
                        if len(Arrive[i]) % 10==0:
                            quot = len(Arrive[i]) // 10
                            for j in range(quot):
                                ten_shipment_id = ship_day.strftime("%Y/%m/%d") + '-' + vdc + '-' + str(delta) + '-' + str(ind[i]) + '-' + str(ind[i] + 101) + '-'+str(j)
                                if len(more_ten_vin[delta]) == 0:
                                    more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                    check_dealer.append(i)
                                    check_dealer_del.append(i)
                                    check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                    if j + 1 == quot:
                                        for key, val in t_ind_name.items():
                                            if val == i:
                                                t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                else:
                                    if i in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_dealer_del.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                    elif i not in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_dealer_del.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                            
                        else:    
                            quot = len(Arrive[i]) // 10
                            for j in range(quot):
                                ten_shipment_id = ship_day.strftime("%Y/%m/%d") + '-' + vdc + '-' + str(delta) + '-' + str(ind[i]) + '-' + str(ind[i] + 101) + '-'+str(j)
                                if len(more_ten_vin[delta]) == 0:
                                    more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                    check_dealer.append(i)
                                    check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                    if j + 1 == quot:
                                        for key, val in t_ind_name.items():
                                            if val == i:
                                                t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                else:
                                    if i in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                    elif i not in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                    else:
                        t_demand[ind[i]] = [len(Arrive[i])]

        
        if len(check_dealer_del)>0:
            set_t_dealer = set(t_dealer)
            set_check_dealer_del = set(check_dealer_del)
            set_dealer = set_t_dealer - set_check_dealer_del
            t_dealer = list(set_dealer)

            t_ind_all = [VDC] + t_dealer
            # print('t_ind_all', t_ind_all)

            t_ind_num = []
            for i in t_ind_all:
                t_ind_num.append(ind[i])
            # print('t_ind_num', t_ind_num)

            first_len = len(t_ind_num)
            # print('first_len', first_len)

            t_ind_dic = {}
            for i in range(len(t_ind_num)):
                t_ind_dic[i] = t_ind_num[i]
            # print('t_ind_dic', t_ind_dic)

            t_ind_name = {}
            for i in range(len(t_ind_num)):
                t_ind_name[i] = t_ind_all[i]
            # print('t_ind_name', t_ind_name)

            t_ind_vin_dic = {}
            t_ind_dealer_dic = {}
            for i in range(1, len(t_ind_name)):
                t_ind_vin_dic[list(t_ind_name.keys())[list(t_ind_name.values()).index(t_ind_name[i])]] = Arrive[
                    t_ind_name[i]]
                t_ind_dealer_dic[list(t_ind_name.keys())[list(t_ind_name.values()).index(t_ind_name[i])]] = t_ind_name[i]
            #     print('t_ind_vin_dic',t_ind_vin_dic)
            #     print('t_ind_dealer_dic',t_ind_dealer_dic)

            t_demand = {}

            test_demand = {}
            for i in t_ind_all:
                str_date = str(day)
                Y = int(str_date[0:4])
                M = int(str_date[5:7])
                D = int(str_date[8:10])
                ship_day = date(Y, M, D)
                # print('####',ship_day)
                # ship_day.strftime("%Y/%m/%d")

                if i == vdc:
                    t_demand[ind[i]] = [0]
                    test_demand[ind[i]] = [0]
                else:
                    if ind[i] not in t_demand:
                        test_demand[ind[i]] = [len(Arrive[i])]
                        if len(Arrive[i]) > 10:
                            t_demand[ind[i]] = [len(Arrive[i]) % 10]
                            if len(Arrive[i]) % 10==0:
                                quot = len(Arrive[i]) // 10
                                for j in range(quot):
                                    ten_shipment_id = ship_day.strftime("%Y/%m/%d") + '-' + vdc + '-' + str(delta) + '-' + str(
                                        ind[i]) + '-' + str(ind[i] + 101) + '-'+str(j)
                                    if len(more_ten_vin[delta]) == 0:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_dealer_del.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                    else:
                                        if i in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                            more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                            check_dealer.append(i)
                                            check_dealer_del.append(i)
                                            check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                            if j + 1 == quot:
                                                for key, val in t_ind_name.items():
                                                    if val == i:
                                                        t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                        elif i not in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                            more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                            check_dealer.append(i)
                                            check_dealer_del.append(i)
                                            check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                            if j + 1 == quot:
                                                for key, val in t_ind_name.items():
                                                    if val == i:
                                                        t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]

                            else:    
                                quot = len(Arrive[i]) // 10
                                for j in range(quot):
                                    ten_shipment_id = ship_day.strftime("%Y/%m/%d") + '-' + vdc + '-' + str(delta) + '-' + str(
                                        ind[i]) + '-' + str(ind[i] + 101) + '-'+str(j)
                                    if len(more_ten_vin[delta]) == 0:
                                        more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                        check_dealer.append(i)
                                        check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                        if j + 1 == quot:
                                            for key, val in t_ind_name.items():
                                                if val == i:
                                                    t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                    else:
                                        if i in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                            more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                            check_dealer.append(i)
                                            check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                            if j + 1 == quot:
                                                for key, val in t_ind_name.items():
                                                    if val == i:
                                                        t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                                        elif i not in check_dealer and Arrive[i][10 * j:10 * (j + 1)] not in check_vin:
                                            more_ten_vin[delta].append((ten_shipment_id, i, Arrive[i][10 * j:10 * (j + 1)]))
                                            check_dealer.append(i)
                                            check_vin.append(Arrive[i][10 * j:10 * (j + 1)])
                                            if j + 1 == quot:
                                                for key, val in t_ind_name.items():
                                                    if val == i:
                                                        t_ind_vin_dic[key] = Arrive[i][10 * (j + 1):]
                        else:
                            t_demand[ind[i]] = [len(Arrive[i])]
        #print('more_ten_vin',more_ten_vin)
        
        #print(len(more_ten_vin[delta]))
        #print('****check_dealer_del',check_dealer_del)    
        min_ten_arrive = []
        loca_list = []
        ten_leg_list = []
        ten_ship_id_list = []
        for i in range(len(more_ten_vin[delta])):
            ten_arrive_time = []
            loca = [vdc]  # ,more_ten_vin[delta][i][1]]
            ten_leg = [
                dist[ind[vdc]][ind[more_ten_vin[delta][i][1]]]]  # ,dist[ind[vdc]][ind[more_ten_vin[delta][i][1]]]]
            ten_ship_id_list.append(more_ten_vin[delta][i][0])
            for j in range(10):
                ten_arrive_time.append(arrive_vin_time[more_ten_vin[delta][i][-1][j]][0])
                loca.append(more_ten_vin[delta][i][1])
                ten_leg.append(0)
            df_min_arrive = pd.DataFrame(ten_arrive_time)
            ten_delivery_start = df_min_arrive[:].max(axis=0)
            min_ten_arrive.append([ten_delivery_start[0]])
            loca_list.append(loca)
            ten_leg_list.append(ten_leg)
        # print(min_ten_arrive)
        # print(loca_list)
        # print('ten_leg_list',ten_leg_list)
        # print(ten_ship_id_list)

        ten_dt_list = []
        for i in range(len(loca_list)):
            ten_dt = []
            ten_s_dt = min_ten_arrive[i][0]
            ten_dt.append(ten_s_dt)
            for j in range(9):
                ten_dt.append(ten_s_dt + timedelta(hours=(dist[ind[vdc]][ind[more_ten_vin[delta][i][1]]] / 30)))
            ten_dt_list.append(ten_dt)

        lead_time_list = []
        lt_list = []
        for i in range(len(loca_list)):
            lead_time = []
            ltlt = []
            lead_t = 0
            # print('*',more_ten_vin[delta][i][2])
            for j in range(10):
                lt = ten_dt_list[i][j] - p_vin_time[more_ten_vin[delta][i][2][j]][0]
                lead_t += lt.total_seconds() / 86400
                ltlt.append(lt.total_seconds() / 86400)
                # print(lead_t)
            lead_time.append(lead_t)
            lt_list.append(ltlt)
            lead_time_list.append(lead_time)
            # print('lead_time_list',lead_time_list)
        # lead_time_dic[delta].append(lead_time_list)

        ten_sol_route_list = []
        for i in range(len(loca_list)):
            ten_sol_route = {"path": [], "vins": [], "cost": []}
            path = []
            vins = []
            check_p = []
            # d_st=delivery_start_list[i]*len(vdc_dealer_list[i])
            for j in zip(loca_list[i], ten_dt_list[i], ten_leg_list[i]):
                if len(path) == 0:
                    path.append((j[0], j[1], j[2]))
                    check_p.append(j[0])
                else:
                    if j[0] not in check_p:
                        path.append((j[0], j[1], j[2]))
                        check_p.append(j[0])
            ten_sol_route['path'] = path
            for k in zip(more_ten_vin[delta][i][2], ten_dt_list[i], lt_list[i]):
                if k[0] == vdc:
                    pass
                else:
                    vins.append((k[0], k[1], k[2]))
            ten_sol_route['vins'] = vins
            ten_cost = lead_time_list[i][0] * 10 + (ten_leg_list[i][0] * 4 + 200)
            ten_sol_route['cost'] = ten_cost

            ten_ship_id = ten_ship_id_list[i]
            ten_sol_route_dic[ten_ship_id] = ten_sol_route
        #         ten_solution_delta[delta].append(ten_sol_route_dic)
        # print(ten_solution_delta)

        #    print('test_demand',test_demand)
        #    print('t_demand',t_demand)

        
        #print(more_ten_vin)
        test_demand_dic = {}
        test_demand_test3 = []
        for i in range(len(t_ind_dic)):
            test_demand_dic[list(t_ind_dic.keys())[list(t_ind_dic.values()).index(t_ind_dic[i])]] = \
                test_demand[t_ind_dic[i]][0]
            test_demand_test3.append(test_demand[t_ind_dic[i]][0])
        # print('test_demand_dic', test_demand_dic)
        # print('test_demand_test3', test_demand_test3)

        demand_dic = {}
        demand_test3 = []
        if len(sol_check_dealer[vdc]) > 0:
            for i in range(len(t_ind_dic)):
                demand_dic[list(t_ind_dic.keys())[list(t_ind_dic.values()).index(t_ind_dic[i])]] = \
                    t_demand[t_ind_dic[i]][0]
                demand_test3.append(t_demand[t_ind_dic[i]][0])
            # print('demand_dic', demand_dic)
            # print('demand_test4', demand_test3)
        else:
            for i in range(len(t_ind_dic)):
                demand_dic[list(t_ind_dic.keys())[list(t_ind_dic.values()).index(t_ind_dic[i])]] = \
                    t_demand[t_ind_dic[i]][0]
                demand_test3.append(t_demand[t_ind_dic[i]][0])
            # print('demand_dic', demand_dic)
            # print('demand_test4', demand_test3)

        data = {}
        _distances = dist

        num_vehicles = 100
        capacities = []
        for i in range(num_vehicles):
            capacities.append(10)

        data["distances"] = _distances
        data['t_ind_dic'] = t_ind_dic
        data["num_locations"] = len(t_ind_all)
        data["num_vehicles"] = num_vehicles
        data["depot"] = 0
        data['t_ind_name'] = t_ind_name
        data['t_ind_all'] = t_ind_all
        data['VDC'] = [VDC]
        data['demand_test'] = demand_test3
        data['vehicle_capacity'] = capacities
        data['t_ind_vin_dic'] = t_ind_vin_dic
        data['t_ind_dealer_dic'] = t_ind_dealer_dic
        data['more_ten_vin'] = more_ten_vin
        data['check_vin'] = check_vin
        data['check_dealer'] = check_dealer
        data['Arrival_time'] = Arrival_time
        data['p_vin_time'] = p_vin_time
        data['p_time'] = p_time
        data['dist'] = dist
        data['ind'] = ind

        return data, ten_solution_delta, ten_ship_id_list

    def create_distance_callback(self, data):

        """Creates callback to return distance between points."""
        distances = data["distances"]
        t_ind_dic = data['t_ind_dic']

        def distance_callback(from_node, to_node):

            if from_node == 0:
                return distances[t_ind_dic[from_node]][t_ind_dic[to_node]] * 4 + 200
            else:
                return distances[t_ind_dic[from_node]][t_ind_dic[to_node]] * 4

        return distance_callback

    def create_demand_callback(self, data):
        """Creates callback to get demands at each location."""

        def demand_callback(from_node, to_node):
            return data['demand_test'][from_node]

        return demand_callback

    def add_distance_dimension(self, routing, distance_callback):
        """Add Global Span constraint"""
        distance = 'Distance'
        maximum_distance = 1000000  # Maximum distance per vehicle.
        routing.AddDimension(
            distance_callback,
            0,  # null slack
            maximum_distance,
            True,  # start cumul to zero
            distance)
        distance_dimension = routing.GetDimensionOrDie(distance)
        # Try to minimize the max distance among vehicles.
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    def add_capacity_constraints(self, routing, data, demand_callback):
        """Adds capacity constraint"""
        capacity = "Capacity"
        routing.AddDimensionWithVehicleCapacity(
            demand_callback,
            0,  # null capacity slack
            data['vehicle_capacity'],  # vehicle maximum capacities
            True,  # start cumul to zero
            capacity)

    def print_solution(self, data, routing, assignment, delta, delta_vins, next_day, sol_check_dealer,
                       period, last_period, cost_delta, min_arrive, delivery_time_dic, lead_time_dic, solution_delta,
                       sol_route_dic, str_date, log=True):

        t_ind_dealer_dic = data['t_ind_dealer_dic']
        t_ind_vin_dic = data['t_ind_vin_dic']
        t_ind_name = data['t_ind_name']
        t_ind_all = data['t_ind_all']

        more_ten_vin = data['more_ten_vin']

        Arrival_time = data['Arrival_time']
        p_vin_time = data['p_vin_time']
        p_time = data['p_time']
        dist = data['dist']
        ind = data['ind']
        vdc = data['VDC'][0]

        total_distance = 0
        g_edge = []
        g_node = []
        for i in t_ind_all:
            if i.isdigit() == True:
                g_node.append(int(i))
            else:
                g_node.append(i)

        count = 0
        sol_test = []
        vdc_vin_list = []
        vdc_dealer_list = []
        ship_id_list = []
        #         sol_route = {"path":[],"vins":[]}
        #         sol_route_dic={}
        #         next_day='2011-01-01'

        Y = int(str_date[0:4])
        M = int(str_date[5:7])
        D = int(str_date[8:10])
        ship_day = date(Y, M, D)
        ship_day.strftime("%Y/%m/%d")

        # print('next_day',next_day)
        for vehicle_id in range(data["num_vehicles"]):
            shipment_id = ship_day.strftime("%Y/%m/%d") + '-' + vdc + '-' + str(delta) + '-' + str(period) + '-' + str(
                vehicle_id)
            route_detail = [shipment_id, next_day]
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)

            route_dist = 0
            route_load = 0
            vdc_vin = []
            vdc_dealer = []

            while not routing.IsEnd(index):

                node_index = routing.IndexToNode(index)
                next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
                if next_node_index != node_index and next_node_index != data["depot"]:
                    if t_ind_name[node_index].isdigit() == True and t_ind_name[next_node_index].isdigit() == True:
                        g_edge.append((int(t_ind_name[node_index]), int(t_ind_name[next_node_index])))
                    elif t_ind_name[node_index].isdigit() == False:
                        g_edge.append((t_ind_name[node_index], int(t_ind_name[next_node_index])))

                if next_node_index != data["depot"]:
                    route_dist += routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)
                route_load += data['demand_test'][node_index]

                if node_index != 0:
                    if next_node_index != 0:
                        if len(t_ind_vin_dic[node_index]) > 10:
                            rr = len(t_ind_vin_dic[node_index]) % 10
                            if rr!=0:
                                plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                               t_ind_dealer_dic[node_index],
                                                                                               t_ind_vin_dic[node_index][
                                                                                               -rr:])
                                route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index][-rr:],
                                                     Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                     routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                                for len_vin in range(len(t_ind_vin_dic[node_index][-rr:])):
                                    #print('rr',rr)
                                    #print(len_vin)
                                    vdc_vin.append(t_ind_vin_dic[node_index][-(len_vin+1)])
                                # append correct number of dealers
                                vdc_dealer.extend([t_ind_dealer_dic[node_index]] * rr)

                            if rr == 0:
                                #print('prob1',len(t_ind_vin_dic[node_index]))
                                plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                               t_ind_dealer_dic[node_index],
                                                                                               t_ind_vin_dic[node_index][
                                                                                               -rr:])
                                route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index][-rr:],
                                                     Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                     routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                                for len_vin in range(len(t_ind_vin_dic[node_index][-rr:])):
                                    vdc_vin.append(t_ind_vin_dic[node_index][-(len_vin+1)])
                                # append correct number of dealers
                                vdc_dealer.extend([t_ind_dealer_dic[node_index]] * rr)
                        else:
                            plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                           t_ind_dealer_dic[node_index],
                                                                                           t_ind_vin_dic[node_index])
                            route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index],
                                                 Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                 routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                            for len_vin in range(len(t_ind_vin_dic[node_index])):
                                vdc_vin.append(t_ind_vin_dic[node_index][len_vin])
                                vdc_dealer.append(t_ind_dealer_dic[node_index])

                    else:
                        if len(t_ind_vin_dic[node_index]) > 10:
                            
                            rr = len(t_ind_vin_dic[node_index]) % 10
                            if rr!=0:
                                plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                               t_ind_dealer_dic[node_index],
                                                                                               t_ind_vin_dic[node_index][
                                                                                               -rr:])
                                route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index][-rr:],
                                                     Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                     routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                                for len_vin in range(len(t_ind_vin_dic[node_index][-rr:])):
                                    vdc_vin.append(t_ind_vin_dic[node_index][-(len_vin+1)])
                                # append correct number of dealers
                                vdc_dealer.extend([t_ind_dealer_dic[node_index]] * rr)

                            if rr == 0:
                                #print('prob2',len(t_ind_vin_dic[node_index]))
                                plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                               t_ind_dealer_dic[node_index],
                                                                                               t_ind_vin_dic[node_index][
                                                                                               -rr:])
                                route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index][-rr:],
                                                     Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                     routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                                for len_vin in range(len(t_ind_vin_dic[node_index][-rr:])):
                                    vdc_vin.append(t_ind_vin_dic[node_index][-(len_vin+1)])
                                # append correct number of dealers
                                vdc_dealer.extend([t_ind_dealer_dic[node_index]] * rr)
                    
                        else:

                            plan_output += ' {0} Load({1}) [dealer:{2},Vin:{3}] ->'.format(node_index, route_load,
                                                                                           t_ind_dealer_dic[node_index],
                                                                                           t_ind_vin_dic[node_index])
                            route_detail.append((t_ind_dealer_dic[node_index], t_ind_vin_dic[node_index],
                                                 Arrival_time[t_ind_dealer_dic[node_index]][-1],
                                                 routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                            for len_vin in range(len(t_ind_vin_dic[node_index])):
                                vdc_vin.append(t_ind_vin_dic[node_index][len_vin])
                                vdc_dealer.append(t_ind_dealer_dic[node_index])
                elif node_index == 0:
                    plan_output += ' {0} Load({1}) ->'.format(node_index, route_load)
                    route_detail.append(
                        (vdc, [], 0, routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)))
                    vdc_vin.append(vdc)
                    vdc_dealer.append(vdc)

                index = assignment.Value(routing.NextVar(index))

            plan_output += ' {}\n'.format(routing.IndexToNode(index))
            plan_output += 'Distance of route: {} miles\n'.format(route_dist)
            plan_output += 'Load of the route: {}\n'.format(route_load)

            route_detail.append(route_load)
            route_detail.append(route_dist)

            if route_load > 0:
                if log: print(plan_output)
                total_distance += route_dist
                count += 1

            if len(vdc_vin) > delta:
                vdc_vin_list.append(vdc_vin)
                ship_id_list.append([shipment_id])
            elif len(vdc_vin) > 1 and last_period:
                vdc_vin_list.append(vdc_vin)
                ship_id_list.append([shipment_id])

            if len(vdc_dealer) > delta:
                # if vdc_dealer
                vdc_dealer_list.append(vdc_dealer)
                # print('0', vdc_dealer_list)
            elif len(vdc_dealer) > 1 and last_period:
                vdc_dealer_list.append(vdc_dealer)

            if len(route_detail) > 5 and route_load >= delta:
                delta_vins[delta].append(route_detail)
                for len_dealer in range(len(route_detail[2:-2])):
                    sol_check_dealer[vdc].append(route_detail[len_dealer + 2][0])

            elif len(route_detail) > 5 and last_period:
                delta_vins[delta].append(route_detail)
                for len_dealer in range(len(route_detail[2:-2])):
                    sol_check_dealer[vdc].append(route_detail[len_dealer + 2][0])

        #     print('delta_vins[delta]',delta_vins[delta])
        #     print('vdc_dealer_list',vdc_dealer_list)
            #print('vdc_vin_list',vdc_vin_list)

        leg_dist_list = []
        for i in range(len(vdc_dealer_list)):
            leg_dist = []
            for j in range(len(vdc_dealer_list[i])):
                #print(j)
                if j != len(vdc_dealer_list[i]) - 1:
                    leg_dist.append(int(dist[ind[vdc_dealer_list[i][j]]][ind[vdc_dealer_list[i][j + 1]]]))
                else:
                    #print(vdc_dealer_list[i][j])
                    #leg_dist.append(int(dist[ind[vdc_dealer_list[i][j]]][ind[vdc_dealer_list[i][j + 1]]]))
                    leg_dist.append(0)
            leg_dist_list.append(leg_dist)
        #     print(leg_dist_list)
            
            

        #     min_arrive={}
        delivery_start_list = []
        for i in range(len(delta_vins[delta])):
            for j in range(len(delta_vins[delta][i]) - 4):
                if delta_vins[delta][i][j + 2][2] != 0:
                    if delta not in min_arrive:
                        min_arrive[delta] = [delta_vins[delta][i][j + 2][2]]
                    else:
                        min_arrive[delta].append(delta_vins[delta][i][j + 2][2])
            #print('min_arrive',min_arrive)
            df_min_arrive = pd.DataFrame(min_arrive[delta])
            delivery_start = df_min_arrive[:].max(axis=0)
            #print('@@@',delivery_start)
            delivery_start_list.append([delivery_start[0]])
            #print('#',delivery_start[0])
        
        #print(1, delivery_start_list)
        #print(2, vdc_dealer_list)
        # print(3, vdc_vin_list)
        
        delivery_time_list = []
        for i in range(len(vdc_dealer_list)):
            delivery_time = []
            start_dt = delivery_start_list[i][0]
            for j in range(len(vdc_dealer_list[i])):
                delivery_time.append(start_dt)
                start_dt += timedelta(hours=(leg_dist_list[i][j] / 30))
            delivery_time_list.append(delivery_time)
        delivery_time_dic[delta].append(delivery_time_list)
        # print('delivery_time_list', delivery_time_list)

        lead_time_list = []
        lt_list = []
        for i in range(len(vdc_vin_list)):
            lead_time = []
            ltlt = []
            lead_t = 0
            for j in range(len(vdc_vin_list[i])):
                if j != 0:
                    lt = delivery_time_list[i][j] - p_vin_time[vdc_vin_list[i][j]][0]
                    lead_t += lt.total_seconds() / 86400
                    ltlt.append(lt.total_seconds() / 86400)
                else:
                    ltlt.append(0)
            lead_time.append(lead_t)
            lt_list.append(ltlt)
            lead_time_list.append(lead_time)

        lead_time_dic[delta].append(lead_time_list)

        total_dist = 0
        total_dist_list = []
        for len_delta in range(len(delta_vins[delta])):
            total_dist += delta_vins[delta][len_delta][-1]
            total_dist_list.append(delta_vins[delta][len_delta][-1])
        # print(total_dist_list)

        sol_route_list = []
        for i in range(len(vdc_dealer_list)):

            sol_route = {}
            path = []
            vins = []
            cost = []
            check_p={}
            check_l=[]
            d_st = delivery_start_list[i] * len(vdc_dealer_list[i])
            
            q=vdc_dealer_list[i]
            w=leg_dist_list[i]
            
            for idx, s in enumerate(zip(q,w)):
                if s[0] not in check_p:
                    check_p[q[idx]] = w[idx]
                else:
                    check_p[q[idx]] = w[idx]
            for j in zip(vdc_dealer_list[i], delivery_time_list[i], leg_dist_list[i]):
                if j[0] not in check_l:
                    path.append((j[0],j[1],check_p[j[0]]))
                    check_l.append(j[0])

            sol_route['path'] = path
            # print(vdc_vin_list[i])
            # print(delivery_time_list[i])
            # print(lt_list[i])
            for k in zip(vdc_vin_list[i], delivery_time_list[i], lt_list[i]):
                if k[0] == vdc:
                    pass
                else:
                    vins.append((k[0], k[1], k[2]))
            sol_route['vins'] = vins

            t_cost = total_dist_list[i] #lead_time_list[i][0] * 10 + total_dist_list[i]
            cost.append(t_cost)
            sol_route['cost'] = cost
            ship_id = ship_id_list[i][0]
            sol_route_dic[ship_id] = sol_route
        #solution_delta[delta].append(sol_route_dic)

        #         print('delivery_time_dic',delivery_time_dic)
        #         print('lead_time_dic',lead_time_dic)
        #         print('delivery_start_list',delivery_start_list)
        #         print('leg_dist_list',leg_dist_list)
        #         print('vdc_vin_list',vdc_vin_list)
        #         print('vdc_dealer_list',vdc_dealer_list)
        #         print('delivery_time',delivery_time)
        #         print('route_detail',route_detail)
        #         print('len_delta_vins',len(delta_vins[delta]))
        #         print('delta_vins',delta_vins)
        #         print('sol_check_dealer',sol_check_dealer[vdc])
        if log: print('Total distance of all routes: {} miles'.format(total_distance))
        if log: print('used vehicle: {}'.format(count))
        # print('more',len(more_ten_vin),'route in this problem',more_ten_vin)
        return g_edge, g_node, lead_time_dic, solution_delta, ship_id_list

    def solve_with_vdc(self, vdc, day, vdc_list, dff2, period, AV_dist, AV=True, log=True,time_out=10):

        start_day = pd.to_datetime(day)
        next_day = start_day + timedelta(days=1)
        last_day = pd.to_datetime(day)
        time_range = last_day - start_day

        delta_vins = {}
        cost_delta = {}
        
        dist = self.dist
        ind = self.ind
        
        min_arrive = {}
        delivery_time_dic = {}
        lead_time_dic = {}
        test_dic = {}
        solution_delta = {}
        ten_solution_delta = {}
        more_ten_vin = {}
        
        start_time=time.time()
        #print('1 - dff2',len(dff2))
        for delta in range(10, 11):
            if start_time-time.time()>time_out:
                continue
            else:
#                 if log:
#                     print('delta', delta)
                delta_vins[delta] = []
                cost_delta[delta] = []
                delivery_time_dic[delta] = []
                lead_time_dic[delta] = []

                full_period = 24
                check_dealer = []
                check_vin = []
                check_dealer_del=[]

                more_ten_vin[delta] = []
                sol_check_dealer = {}

                for i in vdc_list:
                    sol_check_dealer[i] = []
                sol_route_dic = {}
                solution_delta[delta] = sol_route_dic
                ten_sol_route_dic = {}
                ten_solution_delta[delta] = ten_sol_route_dic
                #today_df = dff2.loc[(dff2["arrival_time"] < next_day)]
                #print('today_df',len(dff2))
                
                #print('2 - dff2',len(dff2))
                num_period = period * 2
                
                AV_avail=[]
                AV_df = dff2.loc[AV_avail]
                if AV:
                    #AV_avail=[]
                    for i in zip(dff2['vdc'],dff2['dealer'],dff2['vin']):
                        if dist[ind[i[0]]][ind[i[1]]] < AV_dist:
                            AV_avail.append(i[2])
                    AV_df = dff2.loc[AV_avail]
                    #print('AV delivery',len(AV_avail))
                    #print('3 - dff2',len(dff2))
                    dff2=dff2.drop(AV_avail)
                    #print('3.5 - truck delivery',len(dff2))
                    #print('4 - dff2',len(dff2))

                for i in range(0, num_period):

                    pp = i + 1
                    if log: print(pp)
                    next_day = start_day + timedelta(hours=(full_period / period) * (i + 1))
                    if log: print('next', next_day)
                    day = [str(next_day)]
                    day = day[0][:10]
                    day = pd.to_datetime(day)
                    if (i + 1) % period == 0:
                        day = day - timedelta(days=1)
                    else:
                        day
                    # if log: print('date', day)

                    period_df = dff2.loc[(dff2["arrival_time"] < next_day)]
                    #print('5 - dff2',len(dff2))

                    """Entry point of the program"""
                    # Instantiate the data problem.
                    data, ten_solution_delta, ten_ship_id_list = self.create_data_model(day, period_df, vdc, next_day,
                                                                                        more_ten_vin, check_dealer,
                                                                                        check_vin, sol_check_dealer, delta,
                                                                                        ten_solution_delta,
                                                                                        ten_sol_route_dic,check_dealer_del)
                    
                    
#                     print('%%%%%',more_ten_vin[delta])
#                     print(len(more_ten_vin[delta]))
                    VDC = data['VDC']
                    if log: print('VDC_name', VDC)
                    if log: print("")

                    # Create Routing Model
                    routing = pywrapcp.RoutingModel(data["num_locations"], data["num_vehicles"], data["depot"])

                    # Define weight of each edge
                    distance_callback = self.create_distance_callback(data)
                    demand_callback = self.create_demand_callback(data)
                    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
                    self.add_distance_dimension(routing, distance_callback)
                    self.add_capacity_constraints(routing, data, demand_callback)

                    # Setting first solution heuristic (cheapest addition).
                    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
                    search_parameters.time_limit_ms = 100000
                    search_parameters.first_solution_strategy = (
                        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)  # pylint: disable=no-member
                    # Solve the problem.
                    assignment = routing.SolveWithParameters(search_parameters)
                    if assignment:
                        #                 print('0,ROUTING_NOT_SOLVED: Problem not solved yet\n',
                        #                     '1,ROUTING_SUCCESS: Problem solved successfully\n',
                        #                     '2 ROUTING_FAIL: No solution found to the problem\n',
                        #                     '3 ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution\n',
                        #                     '4 ROUTING_INVALID: Model, model parameters, or flags are not valid\n')
                        #                 print("Solver status: ", routing.status())
                        str_date = str(day)
                        g_edge, g_node, lead_time_dic, solution_delta, ship_id_list = self.print_solution(data, routing,
                                                                                                          assignment, delta,
                                                                                                          delta_vins,
                                                                                                          next_day,
                                                                                                          sol_check_dealer,
                                                                                                          pp,
                                                                                                          pp == num_period,
                                                                                                          cost_delta,
                                                                                                          min_arrive,
                                                                                                          delivery_time_dic,
                                                                                                          lead_time_dic,
                                                                                                          solution_delta,
                                                                                                          sol_route_dic,
                                                                                                         
                                                                                                          str_date, log)

                        if log:
                            G = nx.graph.Graph()

                            for v in self.location.iterrows():
                                if v[1].Location in g_node:
                                    G.add_node(v[1].Location, pos=(
                                        v[1].Latitude, v[1].Longitude if v[1].Longitude > 0 else 360 + v[1].Longitude))

                            for a in g_edge:
                                G.add_edge(a[0], a[1])

                            fig, ax = plt.subplots(1, 1, figsize=(20, 10));
                            nx.draw_networkx(
                                G,
                                pos=nx.get_node_attributes(G, 'pos'),
                                ax=ax,
                                node_size=200,
                                font_size=15,
                                node_shape='o',
                                node_color=['y'])
                            nx.draw_networkx_nodes(
                                G,
                                pos=nx.get_node_attributes(G, 'pos'),
                                nodelist=VDC,
                                node_color='r',
                                node_size=200)
                            # alpha=0.8)

                    else:
                        #                 print('0,ROUTING_NOT_SOLVED: Problem not solved yet\n',
                        #                     '1,ROUTING_SUCCESS: Problem solved successfully\n',
                        #                     '2 ROUTING_FAIL: No solution found to the problem\n',
                        #                     '3 ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution\n',
                        #                     '4 ROUTING_INVALID: Model, model parameters, or flags are not valid\n')
                        #                 print("Solver status: ", routing.status())
                        print('no solution')
                        print('******************************************************************')

        return delta_vins, cost_delta, lead_time_dic, solution_delta, more_ten_vin, ten_solution_delta, ship_id_list, ten_ship_id_list,data,AV_df ,period_df

    def solve_vrp(self, vdc, day, cars_to_deliver_df, period,AV_dist, AV=True, log=True,time_out=10):

        delta_vins, cost_delta, lead_time_dic, solution_delta, more_ten_vin, ten_solution_delta, ship_id_list, ten_ship_id_list,data,AV_df,period_df = self.solve_with_vdc(
            vdc, day, self.vdc_list, cars_to_deliver_df, period,AV_dist, AV, log,time_out=10)
        
        dist = self.dist
        ind = self.ind        
        
        #print(solution_delta)
        if AV:
            period_df = period_df[period_df.vdc.values == vdc]
            dealer_vin_dic = {}
            vin_dealer_dic = {}
            
            dealer_vin_dic = period_df.groupby('dealer')['vin'].apply(list).to_dict()
            vin_dealer_dic = period_df.groupby('vin')['dealer'].apply(list).to_dict()
            #print('######',len(AV_df))
            period_df
            av_route={}
            for i in zip(AV_df['vin'],AV_df['vdc'],AV_df['dealer'],AV_df['arrival_time'],AV_df['pd_time']):
                av_sol_route={}
                str_date = str(i[3])
                Y = int(str_date[0:4])
                M = int(str_date[5:7])
                D = int(str_date[8:10])
                ship_day = date(Y, M, D)
                ship_day.strftime("%Y/%m/%d")
                vin_num = str(i[0])
                av_s_id = ship_day.strftime("%Y/%m/%d")+'-'+str(i[1])+'-'+vin_num[-12:]+'-AV'+'-one'
                av_sol_route['path']=[(i[1],i[3],dist[ind[i[1]]][ind[i[2]]]),(i[2],i[3]+timedelta(hours=(dist[ind[i[1]]][ind[i[2]]] / 30)),0)]
                av_lt = i[3]+timedelta(hours=(dist[ind[i[1]]][ind[i[2]]] / 30))-i[4]
                av_lt=av_lt.total_seconds() / 86400
                av_sol_route['vins']=[(i[0],i[3]+timedelta(hours=(dist[ind[i[1]]][ind[i[2]]] / 30)),av_lt)]
                av_sol_route['cost']=[dist[ind[i[1]]][ind[i[2]]]*0.55]
                av_route[av_s_id] = av_sol_route
                
            av_one_cost = 0
            for av_s_id in av_route:
                av_one_cost += av_route[av_s_id]['cost'][0]
            #print('av_one_cost',av_one_cost)
            #print(av_route)
            
            trans_all={}
            for delta in solution_delta:
                trans_s_cost = 0
                for s_id in solution_delta[delta].keys():
                    route_s_cost=0
                    for q in range(len(solution_delta[delta][s_id]['path'])):
                        route_s_cost += solution_delta[delta][s_id]['path'][q][-1]
                    route_s_cost=route_s_cost*4+200
                    trans_s_cost+=route_s_cost
                #print(trans_s_cost)

                ten_trans_s_cost = 0
                for s_id in ten_solution_delta[delta].keys():
                    ten_route_s_cost=0
                    for q in range(len(ten_solution_delta[delta][s_id]['path'])):
                        ten_route_s_cost += ten_solution_delta[delta][s_id]['path'][q][-1]
                    ten_route_s_cost=ten_route_s_cost*4+200
                    ten_trans_s_cost+=ten_route_s_cost
                    
                trans_all[delta]=trans_s_cost+ten_trans_s_cost
            #print('trans_all',trans_all)
            #print('##########',len(ten_solution_delta[delta]))
            min_trans_delta = min(trans_all.items(), key=lambda x: x[1])
            #        print('min_delta:', min_delta[0])
            try:
                av_solution_delta=self.AV_solution(data,solution_delta,vdc,dealer_vin_dic,vin_dealer_dic)

                av_cost_delta={}
                
                for delta in av_solution_delta:
                    av_sol_cost = 0
                    for s_id in av_solution_delta[delta].keys():
                        av_sol_cost += av_solution_delta[delta][s_id]['cost'][0]
                    #print(av_sol_cost)
                    av_cost_delta[delta]= av_sol_cost 
                #print('av_cost_delta',av_cost_delta) 

                min_av_cost_delta = min(av_cost_delta.items(), key=lambda x: x[1])



                if trans_all[min_trans_delta[0]] > av_cost_delta[min_av_cost_delta[0]]:
                    vdc_cost = [min_av_cost_delta[0],av_cost_delta]
                    solution = av_solution_delta[min_av_cost_delta[0]]
                    solution.update(ten_solution_delta[min_trans_delta[0]])
    #                 print('@@@@@@@@@@@@',av_route)
    #                 print('1',solution)
                    solution.update(av_route)   
    #                 print('2',solution)
                else: 
                    vdc_cost = [min_trans_delta[0], trans_all]
                    #print(solution_delta)
                    solution = solution_delta[min_trans_delta[0]]
                    solution.update(ten_solution_delta[min_trans_delta[0]])
                    solution.update(av_route)
                
            
            except:
                
                vdc_cost = [min_trans_delta[0], trans_all]
                #print(solution_delta)
                solution = solution_delta[min_trans_delta[0]]
                solution.update(ten_solution_delta[min_trans_delta[0]])
                solution.update(av_route)
                
            adjusted_solution = self.adjust_solution(day, solution)
            
            
            
            
        else:
            for delta in solution_delta:
                sol_cost = 0

                for s_id in solution_delta[delta].keys():
                    #                 for id in ship_id_list[i]:
                    sol_cost += solution_delta[delta][s_id]['cost'][0]
                # print(sol_cost)

                ten_sol_cost = 0
                for s_id in ten_solution_delta[delta].keys():
                    #                 for tid in ten_ship_id_list:
                    ten_sol_cost += ten_solution_delta[delta][s_id]['cost']
                # print(ten_sol_cost)

                cost_delta[delta] = sol_cost + ten_sol_cost
            #print('cost_delta',cost_delta)
            #        print('cost_delta', cost_delta)
            min_delta = min(cost_delta.items(), key=lambda x: x[1])
            #        print('min_delta:', min_delta[0])
            vdc_cost = [min_delta[0], cost_delta]
            # solution = delta_vins[min_delta[0]]
            solution = solution_delta[min_delta[0]]
            solution.update(ten_solution_delta[min_delta[0]])
            # print(more_ten_vin[min_delta[0]])
            #print('last_ten_solution_delta',len(ten_solution_delta))


            # make sure the routes departing today
            adjusted_solution = self.adjust_solution(day, solution)

        return vdc_cost, solution, adjusted_solution, solution_delta , ten_solution_delta

    def adjust_solution(self, today, solution):
        today = pd.to_datetime(today)

        depart_times = [
            sol['path'][0][1]
            for s_id, sol 
            in solution.items()
        ]

        adjusted_solution = {}

        for (depart_time, s_id) in zip(depart_times, solution):
            
            # sometimes cost is a list not a value...
            cost = solution[s_id]['cost'] if isinstance(solution[s_id]['cost'], float) else solution[s_id]['cost'][0]
            
            if depart_time < today: # too early
                time_to_delay = today - depart_time
                adjusted_solution[s_id] = {
                    # adjust depart times at all stops
                    'path': [(stop[0], stop[1] + time_to_delay, stop[2]) for stop in solution[s_id]['path']],
                    # adjust arrive times and lead times
                    'vins': [(vin[0], vin[1] + time_to_delay, vin[2] + time_to_delay/timedelta(days=1)) for vin in solution[s_id]['vins']],
                    # add increased leadtime cost
                    'cost': cost + 10 * (time_to_delay/timedelta(days=1)) * len(solution[s_id]['vins'])
                }
            elif depart_time >= today + timedelta(days=1): # tomorrow
                # we just skip this shipment
                pass
            else: # no ajdustment needed
                adjusted_solution[s_id] = {
                    'path': solution[s_id]['path'],
                    'vins': solution[s_id]['vins'],
                    'cost': cost,
                }
                
        return adjusted_solution
    
    def AV_solution(self,data,solution_delta,vdc,dealer_vin_dic,vin_dealer_dic):
        p_vin_time=data['p_vin_time']
        
        dist = self.dist
        ind = self.ind
        
        copy_sol = solution_delta.copy()
        av_solution_delta ={}
        for delta in copy_sol:  
            max_node_list=[]
            new_route={}
            av_route ={}


#             print('vin_dealer',vin_dealer)
            for s_id in copy_sol[delta].keys():
                distance = []
                dealer = []
                delivery = []
                leg_dist = []
                vin=[]
                vin_delivery_time = []
                vin_lead_time=[]
                path = []
                vins = []
                new_sol_route = {}

                av_vin = []
                av_vin_delivery_time = []
                av_vin_lead_time=[]
                av_vins=[]
                #av_route ={}
                #av_sol_route={}

                for i in range(len(copy_sol[delta][s_id]['path'])):
                    dealer.append(copy_sol[delta][s_id]['path'][i][0])
                    from_ = copy_sol[delta][s_id]['path'][0][0]
                    to_=copy_sol[delta][s_id]['path'][i][0]
                    distance.append(dist[ind[from_]][ind[to_]])
                #print(distance)
                #print('dealer',dealer)
                av_distance=[]


                k=0
                for idx, value in enumerate(distance):
                    if value == max(distance):
                        if len(distance)>2 and k==0:
                            max_deal = dealer.pop(idx)
                            k+=1
                        #print('max_deal',max_deal)

                            delivery = [copy_sol[delta][s_id]['path'][0][1]]
                            for j in range(len(dealer)-1):
                                ds = copy_sol[delta][s_id]['path'][0][1]
                                ds +=timedelta(hours=(dist[ind[dealer[j]]][ind[dealer[j+1]]] / 30))
                                delivery.append(ds)

                            for j in range(len(dealer)):
                                if j != len(dealer)-1:
                                    leg_dist.append(dist[ind[dealer[j]]][ind[dealer[j+1]]])
                                else :
                                    leg_dist.append(0)
                #            print(delivery)
                            

                            for i in zip(dealer,delivery,leg_dist):
                                path.append((i[0],i[1],i[2]))

                            new_sol_route['path']=path

                            for j in dealer[1:]:
                                for k in range(len(dealer_vin_dic[j])):
                                    vin.append(dealer_vin_dic[j][k])

                            
                            for j in vin:
                                for p in path:
                                    if vin_dealer_dic[j] == p[0]:
                                        vin_delivery_time.append(p[1])

                            lt_all = 0

                            #p_vin_time[vin[j]]
                            for j in range(len(vin)):
                                lt = vin_delivery_time[j]-p_vin_time[vin[j]][0]
                                lt = lt.total_seconds() / 86400
                                lt_all +=lt
                                vin_lead_time.append(lt)


                            for i in zip(vin,vin_delivery_time,vin_lead_time):
                                vins.append((i[0],i[1],i[2]))
                            new_sol_route['vins']=vins

                            costs = 0
                            for p in path:
                                costs+=p[2]
                           #print('costs',costs)
                            new_sol_route['cost']=[costs*4+200]

                            new_route[s_id]=new_sol_route

                            #for av delivery

                            for j in dealer:
                                av_distance.append(dist[ind[max_deal]][ind[j]])

                            #print(av_distance)
                            q=0
                            av_sol_route={}
                            for idx_, value_ in enumerate(av_distance):
                                if value_ == min(av_distance):
                                    av_min_deal = dealer[idx_]
                                    if max_deal != vdc and av_min_deal != vdc and len(av_distance)>2 and q==0:
                                        #print('%%%',[av_min_deal,max_deal])
                                        for p in path:
                                            if av_min_deal == p[0]:
                                                av_dt=p[1]
                                        #print('max_deal',max_deal)
                                        for k in range(len(dealer_vin_dic[max_deal])):
                                            av_vin.append(dealer_vin_dic[max_deal][k])
                                        #print('av_vin',av_vin)    

                                        for j in av_vin:
                                            av_vin_delivery_time.append(av_dt+timedelta(hours=(dist[ind[av_min_deal]][ind[max_deal]] / 30)))
                                        #print(av_vin_delivery_time)

                                        av_lt_all=0 
                                        for j in range(len(av_vin)):
                                            av_lt = av_vin_delivery_time[j]-p_vin_time[av_vin[j]][0]
                                            av_lt = av_lt.total_seconds() / 86400
                                            av_lt_all+=av_lt
                                            av_vin_lead_time.append(av_lt)


                                        for j in zip(av_vin,av_vin_delivery_time,av_vin_lead_time):
                                            av_vins.append((j[0],j[1],j[2]))
                                        #print(av_min_deal,max_deal)
                                        av_sol_route['path']=[(av_min_deal,av_dt,dist[ind[av_min_deal]][ind[max_deal]]),(max_deal,av_dt +timedelta(hours=(dist[ind[av_min_deal]][ind[max_deal]] / 30)) ,0)]
                                        av_sol_route['vins']=av_vins
                                        av_sol_route['cost']=[dist[ind[av_min_deal]][ind[max_deal]]*0.55]

                                        av_route[s_id+'-AV'] = av_sol_route

                                        q+=1
                        else:
                            new_sol_route['path']=[(dealer[0],copy_sol[delta][s_id]['path'][0][1],dist[ind[dealer[0]]][ind[dealer[1]]]),(dealer[1],copy_sol[delta][s_id]['path'][0][1] +timedelta(hours=(dist[ind[dealer[0]]][ind[dealer[1]]]/30)),0)]
                            new_sol_route['vins']=copy_sol[delta][s_id]['vins']
                            new_sol_route['cost']=[dist[ind[dealer[0]]][ind[dealer[1]]]*0.55]
                            new_route[s_id]=new_sol_route
                        #print(new_route)
                                    


            #print(av_route)
            #print(len(av_route))  
            #print(len(new_route))

            av_solution=new_route
            av_solution.update(av_route)

            #print(len(av_solution))
            av_solution_delta[delta]=av_solution
            #print(av_solution_delta)

        return av_solution_delta

#
# if __name__ == "__main__":
#     location = pd.read_excel('../../Informs_Data_Set/Input_Cost%2C+Location.xlsx')
#     dist = pickle.load(open('../../Exp/dist_mat_9361.dump', 'rb'))
#     ind = pickle.load(open('../../Pickle/location_index_9361.dump', 'rb'))
#     vdc_list = ['3A', '3F', '4J', '7J', '7M', 'BC', 'BE', 'BM', 'CE', 'CW', 'DI', 'DO', 'DV', 'DW', 'DZ', 'EC', 'FF',
#                 'GU', 'JC', 'LM', 'MN', 'MR', 'NM', 'NZ', 'OX', 'PB', 'PH', 'QT', 'QW', 'RJ', 'RO', 'RS', 'RX', 'SO',
#                 'SU', 'SZ', 'UL', 'VE', 'VG', 'VH', 'VW', 'WH', 'WK', 'WL']
#
#     master = pickle.load(
#         open('/Users/YukyungLee/한국외국어대학교/강문정 - informs/Code/yk/2015-01-11.dump', 'rb'))
#
#     # dff = master[~master["next"].isin(vdc_list)]
#
#     vrp = VRP(location, dist, ind, vdc_list)
#
#     vdc = '3A'
#
#     day = '2015-01-11'
#
#     period = 2
#
#     dff = master['vdc_status'][vdc][master['vdc_status'][vdc].delivery == True]
#     # dff = master[~master["next"].isin(vdc_list)]
#     dff2 = dff.sort_values(by='arrival_time')
#
#     cars_to_deliver_df = dff2[
#         (dff2.vdc.values == vdc) &
#         # (dff2.arrival_time >= pd.to_datetime(day)) &
#         (dff2.arrival_time <= pd.to_datetime(day) + timedelta(days=2))
#         ]
#     AV_dist = 200
#     cost, solution, adjusted_solution, solution_delta, ten_solution_delta = vrp.solve_vrp(vdc, day, cars_to_deliver_df,
#                                                                                           period, AV_dist, AV=True,
#                                                                                           log=False, time_out=10)
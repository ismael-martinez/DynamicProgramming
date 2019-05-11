# We are doing a simplified version of Fog Optimization for Dynamic Programming
# This script works with small instances for an exact solution.

import numpy as np
import scipy as sc
from scipy.stats import skewnorm
import pandas as pd
import time
import sys
if not sys.warnoptions:
    import warnings
    from collections import Counter
    warnings.simplefilter("always")
import os
import random
import getopt
from operator import add
from pulp import *

giga_to_mega = 1024 # 1024 MB = 1 GB
FOG_DOWNLOAD = pd.DataFrame([[5.5,1],[11,3],[20,5],[24,5.3], [54, 7], [100, 20], [200, 50], [600, 100], [1300, 250]], columns=['rate', 'var'])
IOT_UPLOAD = pd.DataFrame([[5.5,1],[11,3],[20,5],[24,5.3]], columns=['rate', 'var'])

# Define the state of our system at step k
# Attributes: previous_arrivals (array) - the datapoints in queue, current_storage_available (array) - storage available at each fog device,
#   k (int) - step in our system, cost (float) - The cost choosing optimal decisions from our state onwards
class State:
    def __init__(self, previous_arrivals, current_storage_available, k):
        self.previous_arrivals = previous_arrivals
        self.storage_available = current_storage_available
        self.k = k

# Given the poisson arrival rate parameters, simulate the possible values over N instances
# Input: arrival_rates (array) - The list of n arrival parameters for n devices, N (int) - number of simulated instances per device
# Output: simulated_arrival_rates (list(dict)) - A dictionary to capture the simulated results per device
def simulateArrivalRate(arrival_rates, N):
    simulated_arrivals = [dict()]*len(arrival_rates)
    for i in range(len(arrival_rates)):
        sim_vals = np.random.poisson(lam=arrival_rates[i], size=N).tolist()
        simulated_arrivals_dict = dict()
        for j in sim_vals:
            simulated_arrivals_dict[j] = simulated_arrivals_dict.get(j, 0) + 1
        simulated_arrivals[i] = simulated_arrivals_dict
    return simulated_arrivals

# Calculates the distance between every IoT Device and every Fog Device into a matrix
# Input: IoT_devices (DataFrame) - set of IoT Devices, fog_devices (DataFrame) - set of fog devices
# Output: distanceMatrix (np 2d array) - matrix of distances from iot (rows) to fog (cols)
def distanceMatrix(IoT_devices, fog_devices):
    distanceMatrix = np.empty([IoT_devices.shape[0], fog_devices.shape[0]])

    for i in range(distanceMatrix.shape[0]):
        for j in range(distanceMatrix.shape[1]):
            distanceMatrix[i, j] = np.sqrt((fog_devices.loc[j, 'xloc'] - IoT_devices.loc[i, 'xloc'])**2 + (fog_devices.loc[j, 'yloc'] + IoT_devices.loc[i, 'yloc'])**2)
    return distanceMatrix

# In order to make the most effective choices in a reasonable order, we rank the fog devices by latency cost and transmission cost from IoT to Fogs
# Input: distanceMatrix (np 2darray) - latency of IoT i to fog j, df_fog (DataFrame) - DataFrame of fog devices including download_rate, mean_file_size (float) - average of the file size to calculate transfer rate
# Output: fog_rank (DataFrame) - Ascending order of most lucrative decisions, on average, with their value
def rankFogDevices(distanceMatrix, df_fog, mean_file_size):
    mean_latency = np.mean(distanceMatrix, axis=0)
    fog_rank = pd.DataFrame(index=range(len(mean_latency)), columns=['avg_cost'])
    fog_rank = fog_rank.join(df_fog)
    del df_fog
    for id in range(len(mean_latency)):
        fog_rank.loc[id, 'avg_cost'] = mean_latency[id] + mean_file_size/fog_rank.loc[id, 'download_rate']
    fog_rank = fog_rank.sort_values(by='avg_cost').reset_index(drop=True)
    return fog_rank

# In order to make the most effective choices in a reasonable order, we rank the IoT devices by file size
# Input: df_IoT (DataFrame) - DataFrame of IoT devices including file sizes and arrival_means
# Output: IoT_rank (DataFrame) - Ascending order of the most lucrative decisions, on average, with their value
def rankIoTDevices(df_IoT):
    df_IoT['id'] = df_IoT.index
    df_IoT['mean_file_size'] = df_IoT['file_size'].multiply(df_IoT['poisson_arrival_rate'])
    df_IoT = df_IoT.sort_values(by='mean_file_size')
    return df_IoT

# Based on the current state, what is the best decision to take
# Input: df_IoT (DataFrame) - Set of IoT Devices, df_fog (DataFrame) - set of fog devices,
#   distanceMatrix (np 2d array) - Latency matrix for device i to fog j, datapoints (array) - list of arriving datapoints
# Output: optimal_decisions (set) - Set of possible decisions, without repetition
def bestKDecision(df_IoT, df_fog, distanceMatrix, datapoints, k=25):
    df_fog_copy = df_fog.copy()
    df_fog = df_fog_copy
    optimal_decisions = set()
    fog_rank = rankFogDevices(distanceMatrix, df_fog, 25)['id']
    # Calculate Upload time
    IoT_transfer_rate = pd.DataFrame(df_IoT.loc[:, ['file_size', 'upload_rate']]).join(pd.DataFrame({'datapoints':datapoints}))
    IoT_transfer_rate['package_size'] = IoT_transfer_rate['file_size'].multiply(IoT_transfer_rate['datapoints'])
    IoT_transfer_rate['upload_time'] = IoT_transfer_rate['package_size'].divide(IoT_transfer_rate['upload_rate'])
    # Calculate download time matrix
    download_time_matrix = np.empty([df_IoT.shape[0], df_fog.shape[0]])
    for i in range(download_time_matrix.shape[0]):
        for j in range(download_time_matrix.shape[1]):
            download_time_matrix[i, j] = IoT_transfer_rate.loc[i, 'package_size']/df_fog.loc[j, 'download_rate']

    total_cost_matrix = 2*distanceMatrix + download_time_matrix
    upload_array = IoT_transfer_rate['upload_time'].as_matrix().reshape([df_IoT.shape[0],1])
    total_cost_matrix = total_cost_matrix + upload_array

    package_size = IoT_transfer_rate.loc[:, ['package_size']]

    del IoT_transfer_rate
    del download_time_matrix
    del upload_array
    del df_IoT

    package_size = package_size.sort_values(by='package_size').reset_index()
    possibilities = 2*package_size.shape[0]
    storage_possibilities = [0]*(possibilities)
    low = 0
    high = sum(package_size['package_size'])
    for i in range(int(possibilities/2)):
        storage_possibilities[i] = low
        storage_possibilities[possibilities-1-i] = high
        low += package_size.loc[i, 'package_size']
        high -= package_size.loc[i, 'package_size']
    storage_possibilities.sort()
    storage_delta = [0] + [j - i for i, j in zip(storage_possibilities[:-1], storage_possibilities[1:])] # or use itertools.izip in py2k

    storageAvailable = df_fog.loc[:, 'storageLeft']

    for index, storage in storageAvailable.iteritems():
        if index != 0:
            storageAvailable.at[index] = np.min([storage, sum(package_size['package_size'])])

    del df_fog
    latency_matrix_size = total_cost_matrix.shape
    for run in range((latency_matrix_size[1]-1)*possibilities):
        fog_iteration = int(np.floor(run/possibilities))
        storage_possibility = run % possibilities
        if storage_possibility == 5 and fog_iteration < latency_matrix_size[1]-2:
            continue
        for j in range(1, latency_matrix_size[1]):
            if fog_rank[j] < fog_iteration:
                storageAvailable.at[j] = 0
            elif fog_rank[j] == fog_iteration:
                storageAvailable.at[j] -= storage_delta[storage_possibility]
                storageAvailable.at[j] = np.max([0, storageAvailable.at[j]])
        # Optimize for decision that minimizes cost subject to storage constraints of Fog - use PuLP
        prob = LpProblem('LatencyMin', LpMinimize)
            # Decision Variables
        x = [0]*(total_cost_matrix.size) # c_i,j pertains to x[(i-1)*F + (j-1)]

        for i in range(latency_matrix_size[0]):
            for j in range(latency_matrix_size[1]):
                index = (i)*latency_matrix_size[1] + j
                x[index] = LpVariable("Cost (%d,%d)" % (i,j), 0, 1, LpInteger)
        for l, i in zip(np.nditer(total_cost_matrix), range(total_cost_matrix.size)):
            if i != 0:
                lp_objective += x[i]*l
            else:
                lp_objective = x[i]*l
        prob += lp_objective, "optimal latency decision"
            # Contrainsts
                # Memory storage constraint
        for j in range(1, latency_matrix_size[1]):
            index = j
            package_contraint = x[index]*package_size['package_size'][0]
            for i in range(1, latency_matrix_size[0]):
                index = i*latency_matrix_size[1] + j
                package_contraint += x[index]* package_size['package_size'][i]
            prob += package_contraint <= storageAvailable[j], "Sum of IoT packages must fit in storage M_%d" % j
                # Sum of device choices equals 1
        for i in range(latency_matrix_size[0]):
            index = i*latency_matrix_size[1]
            sum_constraint = x[index]
            for j in range(1, latency_matrix_size[1]):
                index = i*latency_matrix_size[1] + j
                sum_constraint += x[index]
            prob += sum_constraint == 1, "Sum of decisions from IoT %d must equal 1" % i


        prob.solve()
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] != 'Optimal':
            warnings.warn("Couldn't reach an optimal solution.")


        decisions = [-1]*latency_matrix_size[0]
        for v in prob.variables():
            if v.varValue == 1:
                #parse v.name
                ij = v.name.split('(')[1].split(')')[0].split(',')
                decisions[int(ij[0])] = int(ij[1])

        optimal_decisions.add(tuple(decisions))
        if len(optimal_decisions) >= k:
            break

    return optimal_decisions

# We want to distribute the D IoT data packets to the F devices. Determine if a defined decision is valid
# Input: IoTDevices (DataFrame) - set of IoT devices, fogDevices (DataFrame) - set of Fog devices, decision (array) - size D array of decisions, send package d (IoT) to which fog id
# Output: valid.all() (boolean) - Whether the fogs have enough space to fit the packages, valid (array of boolean) - list of which values failed
def validDecision(IoTDevices, fogDevices, decision, datapoints):
    df_decision = pd.DataFrame({'decision': decision})
    df_datapoints = pd.DataFrame({'datapoints': datapoints})
    package = pd.DataFrame(IoTDevices.loc[:, ['id', 'file_size', 'upload_rate']]).join(df_datapoints)
    package['full_data_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
    package = package.join(df_decision)
    package = package.groupby('decision')
    package_by_fog = package.sum()
    del package
    fog_memory = pd.DataFrame(fogDevices.loc[:, 'storageLeft'])
    memory_comparison = pd.merge(fog_memory, package_by_fog,left_index=True, right_on='decision')
    valid = memory_comparison['storageLeft'] >= memory_comparison['full_data_size']
    return valid.all()

# After validation, calculate the cost of a decision. There is no stochasticity involved in this cost since it is involved in the inputs
# Input: IoTDevices (DataFrame) - set of devices, fogDevices (DataFrame) - set of fog devices,
#   distance_matrix (np 2d array) - matrix of distances from IoT to fog, decision (array) - size D array of decision, send package d (IoT) to which fog id
def decisionCost(IoTDevices, fogDevices, distance_matrix, decision, datapoints, state, cost_table, data_arrival_next_period):
    df_decision = pd.DataFrame({'decision': decision})
    df_datapoints = pd.DataFrame({'datapoints': datapoints})
    package = pd.DataFrame(IoTDevices.loc[:, ['id', 'file_size', 'upload_rate']]).join(df_datapoints)
    package['full_data_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
    package = package.join(df_decision)
    package = pd.merge(package, pd.DataFrame(fogDevices.loc[:, ['id','download_rate', 'storageLeft']]), left_on='decision', right_on='id', suffixes=('_D', '_F'))
    package = package.sort_values(by='id_D').reset_index()
    package['latency'] = distance_matrix[package.loc[:, 'id_D'], package.loc[:, 'id_F']]
    #print(package.loc[:, ['full_data_size', 'upload_rate', 'download_rate', 'latency']])

    cost = package['full_data_size'].divide(package['upload_rate']).add(
        package['full_data_size'].divide(package['download_rate'])).add(
        2*package['latency']).sum()
    next_state, fogDevices = transitionToNextState(IoTDevices, fogDevices, state, decision, state.previous_arrivals, data_arrival_next_period, state.k)
    next_state_cost = 0
    next_state_index = 0
    for index, row in cost_table.iterrows():
        if row['k'] == next_state.k:
            if list(map(add,row['storage_left'], [1]*len(state.storage_available))) >= next_state.storage_available and \
                    list(map(add,row['storage_left'], [-1]*len(state.storage_available))) <= next_state.storage_available:
                if next_state_cost < row['cost']:
                    next_state_cost = row['cost']
                    next_state_index = index
    #print(cost)
    #print(next_state_cost)
    return [cost + next_state_cost, next_state_index]

# Given a current state, a decision and a random poisson arrival of data, we define the next state
# Input: df_IoT (DataFrame) - Description of IoT Devices, df_fog (DataFrame) - Description of Fog Devices,
#   decision (array) - Disbursement decision of IoT to fog devices, data_arrival_last_period (array) - used to calculate package size per IoT Device
#   data_arrival_next_period (array) - list of how many datapoints per IoT device arrived in the last period
# Output: data_arrival_next_period (array) - list of how many datapoints per IoT device arrived in the last period,
#   df_fog_updated (DataFrame) - The storage left of each fog device is updated per the decision
def transitionToNextState(df_IoT, df_fog, state, decision, data_arrival_last_period, data_arrival_next_period, k):
    df_decision = pd.DataFrame(df_IoT.loc[:, 'file_size']).join(pd.DataFrame({'datapoints': data_arrival_last_period}))
    df_decision['package_size'] = df_decision['file_size'].multiply(df_decision['datapoints'])
    df_decision = df_decision.join(pd.DataFrame({'decision':decision}))
    df_decision = df_decision.groupby('decision')
    df_storage_needed = df_decision.sum()
    df_fog_updated = df_fog.copy()
    for index, row in df_fog_updated.iterrows():
        df_fog_updated.loc[index, 'storageLeft'] = state.storage_available[index]
    for index, row in df_storage_needed.iterrows():
        df_fog_updated.loc[index, 'storageLeft'] -= row['package_size']
    state = State(data_arrival_next_period, df_fog_updated.loc[:, 'storageLeft'].tolist(), k=k+1)

    return [state, df_fog_updated]


# Compare different expected costs and choose the minimum cost
def DynamicProgramming(IoTDevices, fogDevices, distance_matrix, arrivals, N):

    # Build the network of states for the last two layers only
    if N == 1:
        sys.exit("N must be greater than 1")
    storage_init = fogDevices.loc[:, 'storageLeft'].copy().tolist()
    for i in range(len(storage_init)):
        if storage_init[i] != np.inf:
            storage_init[i] = 0.0
    final_state = State([0]*IoTDevices.shape[0], storage_init, k=N)
    cost_table = pd.DataFrame([[final_state.storage_available, final_state.previous_arrivals, N, 0, 'root']], columns=['storage_left', 'previous_arrival', 'k', 'cost', 'successor'])

    decision_datapoints = [1,2, 1] # TODO Determine all possible datapoints
    next_period_datapoints = [0, 0, 0]


    decision_set = bestKDecision(IoTDevices, fogDevices, distanceMatrix=distance_matrix, datapoints=decision_datapoints)
        # Create a set of States based on possible decisions and the next state
    for k in range(N-1, -1, -1):
        next_state = cost_table.where(cost_table['k'] == k+1)
        next_state = next_state.dropna()
        next_state_storage =  next_state['storage_left'].tolist()
        next_state_index = next_state.index.tolist()
        states = []

        for idx in range(len(next_state_storage)):
            storage = next_state_storage[idx]
            ct_index = next_state_index[idx]
            for decision in decision_set:
                storage_state = storage.copy()
                decision = list(decision)
                package_size = pd.DataFrame(IoTDevices.loc[:, 'file_size']).join(pd.DataFrame({'decision': decision})).join(pd.DataFrame({'datapoints':decision_datapoints}))
                package_size['package_size'] = package_size['file_size'].multiply(package_size['datapoints'])
                packages_sent = package_size.loc[:, ['decision', 'package_size']].groupby('decision')
                packages_sent = packages_sent.sum()
                for index, row in packages_sent.iterrows():
                    storage_state[index] += row['package_size']
                state = State(previous_arrivals=decision_datapoints, current_storage_available=storage_state, k=k)
                states.append(state)


        for state in states:
            min_cost = np.inf
            next_state_index = None
            argmin_cost = None
            for decision in decision_set:
                decision = list(decision)
                fogDevicesReducedStorage = fogDevices.copy()
                fogDevicesStorage = pd.DataFrame({'storageLeft': state.storage_available})
                fogDevicesReducedStorage.loc[:, 'storageLeft'] = fogDevicesStorage
                #print(fogDevicesReducedStorage['storageLeft'])
                if validDecision(IoTDevices, fogDevicesReducedStorage, decision, decision_datapoints):
                    [expected_cost, cost_index] = decisionCost(IoTDevices, fogDevices, distance_matrix, decision, decision_datapoints, state, cost_table, next_period_datapoints)
                    if expected_cost < min_cost:
                        min_cost = expected_cost
                        next_state_index = cost_index
                        argmin_cost = decision

            state_cost_df = pd.DataFrame([[state.storage_available, state.previous_arrivals, k, min_cost, next_state_index]], columns=['storage_left', 'previous_arrival', 'k', 'cost', 'successor'])
            cost_table = cost_table.append(state_cost_df, ignore_index=True)


            #print(state_cost_df)
            #print(cost_table.loc[:, ['storage_left', 'cost', 'k', 'successor']])
    print(cost_table)
    cost_table.to_csv('optimal_solution.csv')
    return 0

# Main function of FogAssignment Problem
# Assign IoT tasks to Fog devices based on computation power, time, and congestion
def main(D_file, F_file, sd, N):
    np.random.seed(seed=sd)

    try:
        print(F_file)
        fogs = pd.read_csv(F_file)
        devices = pd.read_csv(D_file)

    except:
        sys.exit("Was not able to read in Fog system with the given parameters")

    dm = distanceMatrix(devices, fogs)
    simulated_arrival = simulateArrivalRate(devices.loc[:, 'poisson_arrival_rate'].tolist(), 100)
    DynamicProgramming(devices, fogs, dm, simulated_arrival, N=N)

def usage():
    print("-h Help\n-d csv file path to IoT device data\n-f csv file path to Fog device data\n-N Number of steps\n-s Random seed (optional)")

if __name__ == '__main__':
    # Start timer
    start = time.time()
    device_num = 0
    fog_num = 0
    N = 0
    seed = None
    debug = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:f:s:N:")
    except getopt.GetoptError as err:
        usage()
        sys.exit('The command line inputs were not given properly')
    for opt, arg in opts:
        if opt == '-d':
            device_num = arg
        elif opt == '-f':
            fog_num = arg
        elif opt == '-s':
            seed = int(arg)
        elif opt == '-N':
            N = int(arg)
        else:
            usage()
            sys.exit(2)
    if device_num == 0 or fog_num == 0 or N == 0:
        usage()
        sys.exit(2)

    main(device_num, fog_num, seed, N)
    print("Time in sec: " + str(time.time() - start))

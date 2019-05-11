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
import itertools

giga_to_mega = 1024 # 1024 MB = 1 GB
FOG_DOWNLOAD = pd.DataFrame([[5.5,1],[11,3],[20,5],[24,5.3], [54, 7], [100, 20], [200, 50], [600, 100], [1300, 250]], columns=['rate', 'var'])
IOT_UPLOAD = pd.DataFrame([[5.5,1],[11,3],[20,5],[24,5.3]], columns=['rate', 'var'])
COST_TABLE_COLUMNS = ['cost', 'storage_available', 'previous_arrivals', 'k', 'successor']

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
    simulated_arrivals_cp = [dict()]*len(arrival_rates)
    for i in range(len(arrival_rates)):
        sim_vals = np.random.poisson(lam=arrival_rates[i], size=N).tolist()
        simulated_arrivals_dict = dict()
        for j in sim_vals:
            simulated_arrivals_dict[j] = simulated_arrivals_dict.get(j, 0) + 1
        simulated_arrivals[i] = simulated_arrivals_dict
        simulated_arrivals_cp[i] = simulated_arrivals_dict.copy()
    for idx in range(len(simulated_arrivals)): # Reduce the size of the simulated results
        for key in simulated_arrivals[idx]:
            if simulated_arrivals[idx][key] < 10:
                simulated_arrivals_cp[idx].pop(key, None)
    return simulated_arrivals

# Based on a set of attributes, calculate a set of values J_k to tune parameters r
# Input: sim_arrivals (list(dict)) - Set of simulated number of arrivals for each device, latency_matrix (np 2darray) - latency matrix of device i to fog j
    # df_IoT (DataFrame) - IoT device data, df_fog (DataFrame) - Fog device data, N (int) - total number of steps
# Ouptut: r (np array) - Tune parameters based on feature vector
def baseApproximationTune(sim_arrivals, latency_matrix, df_IoT, df_fog, N):
    # Create a table of costs
    C = 10

    return 0

# Based on a set of attributes, calculate a base approximation of the cost to the end
# Input: State (State) - State variable of the current system, latency_matrix (np 2darray) - latency matrix of device i to fog j
    # df_IoT (DataFrame) - IoT device data, df_fog (DataFrame) - Fog device data
# Ouput: base_cost (float) - Approximation cost
def baseApproximationCalculate(State, latency_matrix, df_IoT, df_fog):
    k = State.k
    storage_available = State.storage_available
    previous_arrival = State.previous_arrivals

    return 10 # TODO

# Calculates the distance between every IoT Device and every Fog Device into a matrix
# Input: IoT_devices (DataFrame) - set of IoT Devices, fog_devices (DataFrame) - set of fog devices
# Output: distanceMatrix (np 2d array) - matrix of distances from iot (rows) to fog (cols)
def distanceMatrix(df_IoT, df_fog):
    distanceMatrix = np.empty([df_IoT.shape[0], df_fog.shape[0]])

    for i in range(distanceMatrix.shape[0]):
        for j in range(distanceMatrix.shape[1]):
            distanceMatrix[i, j] = np.sqrt((df_fog.loc[j, 'xloc'] - df_IoT.loc[i, 'xloc']) ** 2 + (df_fog.loc[j, 'yloc'] + df_IoT.loc[i, 'yloc']) ** 2)
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



# We want to distribute the D IoT data packets to the F devices. Determine if a defined decision is valid
# Input: IoTDevices (DataFrame) - set of IoT devices, fogDevices (DataFrame) - set of Fog devices, decision (array) - size D array of decisions, send package d (IoT) to which fog id
# Output: valid.all() (boolean) - Whether the fogs have enough space to fit the packages, valid (array of boolean) - list of which values failed
def validDecision(df_IoT, df_fog, decision, datapoints):
    package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size', 'upload_rate']]).join(pd.DataFrame({'datapoints': datapoints}))
    package['package_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
    package = package.join(pd.DataFrame({'decision': decision}))
    package = package.groupby('decision')
    package = package.sum()
    fog_memory = pd.DataFrame(df_fog.loc[:, 'storageLeft'])
    memory_comparison = pd.merge(fog_memory, package,left_index=True, right_on='decision')
    valid = memory_comparison['storageLeft'] >= memory_comparison['package_size']
    return valid.all()

# After validation, calculate the cost of a decision. There is no stochasticity involved in this cost since it is involved in the inputs
# Input: IoTDevices (DataFrame) - set of devices, fogDevices (DataFrame) - set of fog devices,
#   distance_matrix (np 2d array) - matrix of distances from IoT to fog, decision (array) - size D array of decision, send package d (IoT) to which fog id,
#   datapoints (list) - the previous arrival datapoints, state (State) - the current state of the system,
#   data_arrival_next_period (list)_ - the simulated amount of data to arrive next period,arrivals (list(dict)) - simulation results of arrivals
def decisionCostHat(df_IoT, df_fog, distance_matrix, decision, datapoints, state, data_arrival_next_period, arrivals):
    package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size', 'upload_rate']]).join(pd.DataFrame({'datapoints': datapoints}))
    package['package_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
    package = package.join(pd.DataFrame({'decision': decision}))
    package = package.groupby('decision')
    package = package.sum()
    package = pd.merge(package, pd.DataFrame(df_fog.loc[:, ['id', 'download_rate', 'storageLeft']]), left_on='decision', right_on='id', suffixes=('_D', '_F'))
    package = package.sort_values(by='id_D').reset_index()
    package['latency'] = distance_matrix[package.loc[:, 'id_D'], package.loc[:, 'id_F']]
   # print(package.loc[:, ['package_size', 'upload_rate', 'download_rate', 'latency']])

    cost = package['package_size'].divide(package['upload_rate']).add(
        package['package_size'].divide(package['download_rate'])).add(
        2*package['latency']).sum()
    next_state, df_fog = transitionToNextState(df_IoT, df_fog, state, decision, state.previous_arrivals, data_arrival_next_period, state.k)
    next_state_cost = baseApproximationCalculate(state, distance_matrix, df_IoT, df_fog)

    return cost + next_state_cost

# Given a current state, a decision and a random poisson arrival of data, we define the next state
# Input: df_IoT (DataFrame) - Description of IoT Devices, df_fog (DataFrame) - Description of Fog Devices,
#   decision (array) - Disbursement decision of IoT to fog devices, data_arrival_last_period (array) - used to calculate package size per IoT Device
#   data_arrival_next_period (array) - list of how many datapoints per IoT device arrived in the last period
# Output: data_arrival_next_period (array) - list of how many datapoints per IoT device arrived in the last period,
#   df_fog_updated (DataFrame) - The storage left of each fog device is updated per the decision
def transitionToNextState(df_IoT, df_fog, state, decision, data_arrival_last_period, data_arrival_next_period, k):
    package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size']]).join(
        pd.DataFrame({'datapoints': data_arrival_last_period}))
    package['package_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
    package = package.join(pd.DataFrame({'decision': decision}))
    package = package.groupby('decision')
    package = package.sum()
    df_fog_updated = df_fog.copy()
    for index, row in df_fog_updated.iterrows():
        df_fog_updated.loc[index, 'storageLeft'] = state.storage_available[index]
    for index, row in package.iterrows():
        df_fog_updated.loc[index, 'storageLeft'] -= row['package_size']
    state = State(data_arrival_next_period, df_fog_updated.loc[:, 'storageLeft'].tolist(), k=k+1)

    return [state, df_fog_updated]

# Compare different expected costs and choose the minimum cost
def DynamicProgramming(df_IoT, df_fog, distance_matrix, arrivals, N):

    # Build the network of states for the last two layers only
    if N == 1:
        sys.exit("N must be greater than 1")
    # Calculate latency matrix, simulate arrival of data, all possible decisions, and set final state
    latency_matrix = distanceMatrix(df_IoT, df_fog)
    #simulateArrivalRate(arrivals, N)


    arrivals_list = []
    for d in arrivals:
        arrivals_list.append([key for key in d])
    arrival_set = set([arrival for arrival in list(itertools.product(arrivals_list[0], arrivals_list[1]))])
    storage_init = df_fog.loc[:, 'storageLeft'].copy().tolist()
    for i in range(len(storage_init)):
        if storage_init[i] != np.inf:
            storage_init[i] = 0.0
    final_state = State([0] * df_IoT.shape[0], storage_init, k=N)
    cost_table = pd.DataFrame(columns=COST_TABLE_COLUMNS)

    cost_table = cost_table.append(pd.DataFrame([[0, final_state.storage_available, final_state.previous_arrivals, N, 'root']], columns=COST_TABLE_COLUMNS))

    decision_set = set([decision for decision in itertools.product(list(range(1, df_fog.shape[0])), repeat=df_IoT.shape[0])])
    datapoints = [1,2]

    for k in range(N-1, -1, -1):
        print(k)

        next_states = cost_table[cost_table.loc[:, 'k'] == k+1]
        print(datapoints)
        for st_index, next_state in next_states.iterrows():
            for decision in decision_set:
                # TODO OR, look at how many rounds k we can do within the set memory
                storage_available = next_state.storage_available.copy()
                package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size', 'upload_rate']]).join(
                    pd.DataFrame({'datapoints': datapoints}))
                package['package_size'] = package.loc[:, 'file_size'] * package.loc[:, 'datapoints']
                package = package.join(pd.DataFrame({'decision': decision}))
                package = package.groupby('decision')
                package = package.sum()
                for index, row in package.iterrows():
                    storage_available[index] += row['package_size']
                state = State(datapoints, storage_available, k)
                if k == 1:
                    cost = decisionCostHat(df_IoT, df_fog, distance_matrix, decision, datapoints, state, [0,0], arrivals)
                else:
                    cost = decisionCostHat(df_IoT, df_fog, distance_matrix, decision, datapoints, state, datapoints, arrivals)
                cost_table = cost_table.append(
                    pd.DataFrame([[cost, state.storage_available, state.previous_arrivals, k, st_index]],
                                 columns=COST_TABLE_COLUMNS), ignore_index=True)

    print(cost_table)


    return 0

# Main function of FogAssignment Problem
# Assign IoT tasks to Fog devices based on computation power, time, and congestion
def main(D_file, F_file, sd, N):
    np.random.seed(seed=sd)

    try:
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

# Based on the current state, what is the best decision to take
# # Input: df_IoT (DataFrame) - Set of IoT Devices, df_fog (DataFrame) - set of fog devices,
# #   distanceMatrix (np 2d array) - Latency matrix for device i to fog j, datapoints (array) - list of arriving datapoints
# # Output: optimal_decisions (set) - Set of possible decisions, without repetition
# def bestKDecision(df_IoT, df_fog, distanceMatrix, datapoints, k=25):
#     df_fog_copy = df_fog.copy()
#     df_fog = df_fog_copy
#     optimal_decisions = set()
#     fog_rank = rankFogDevices(distanceMatrix, df_fog, 25)['id']
#     # Calculate Upload time
#     IoT_transfer_rate = pd.DataFrame(df_IoT.loc[:, ['file_size', 'upload_rate']]).join(pd.DataFrame({'datapoints':datapoints}))
#     IoT_transfer_rate['package_size'] = IoT_transfer_rate['file_size'].multiply(IoT_transfer_rate['datapoints'])
#     IoT_transfer_rate['upload_time'] = IoT_transfer_rate['package_size'].divide(IoT_transfer_rate['upload_rate'])
#     # Calculate download time matrix
#     download_time_matrix = np.empty([df_IoT.shape[0], df_fog.shape[0]])
#     for i in range(download_time_matrix.shape[0]):
#         for j in range(download_time_matrix.shape[1]):
#             download_time_matrix[i, j] = IoT_transfer_rate.loc[i, 'package_size']/df_fog.loc[j, 'download_rate']
#
#     total_cost_matrix = 2*distanceMatrix + download_time_matrix
#     upload_array = IoT_transfer_rate['upload_time'].as_matrix().reshape([df_IoT.shape[0],1])
#     total_cost_matrix = total_cost_matrix + upload_array
#
#     package_size = IoT_transfer_rate.loc[:, ['package_size']]
#
#     del IoT_transfer_rate
#     del download_time_matrix
#     del upload_array
#     del df_IoT
#
#     package_size = package_size.sort_values(by='package_size').reset_index()
#     possibilities = 2*package_size.shape[0]
#     storage_possibilities = [0]*(possibilities)
#     low = 0
#     high = sum(package_size['package_size'])
#     for i in range(int(possibilities/2)):
#         storage_possibilities[i] = low
#         storage_possibilities[possibilities-1-i] = high
#         low += package_size.loc[i, 'package_size']
#         high -= package_size.loc[i, 'package_size']
#     storage_possibilities.sort()
#     storage_delta = [0] + [j - i for i, j in zip(storage_possibilities[:-1], storage_possibilities[1:])] # or use itertools.izip in py2k
#
#     storageAvailable = df_fog.loc[:, 'storageLeft']
#
#     for index, storage in storageAvailable.iteritems():
#         if index != 0:
#             storageAvailable.at[index] = np.min([storage, sum(package_size['package_size'])])
#
#     del df_fog
#     latency_matrix_size = total_cost_matrix.shape
#     for run in range((latency_matrix_size[1]-1)*possibilities):
#         fog_iteration = int(np.floor(run/possibilities))
#         storage_possibility = run % possibilities
#         if storage_possibility == 5 and fog_iteration < latency_matrix_size[1]-2:
#             continue
#         for j in range(1, latency_matrix_size[1]):
#             if fog_rank[j] < fog_iteration:
#                 storageAvailable.at[j] = 0
#             elif fog_rank[j] == fog_iteration:
#                 storageAvailable.at[j] -= storage_delta[storage_possibility]
#                 storageAvailable.at[j] = np.max([0, storageAvailable.at[j]])
#         # Optimize for decision that minimizes cost subject to storage constraints of Fog - use PuLP
#         prob = LpProblem('LatencyMin', LpMinimize)
#             # Decision Variables
#         x = [0]*(total_cost_matrix.size) # c_i,j pertains to x[(i-1)*F + (j-1)]
#
#         for i in range(latency_matrix_size[0]):
#             for j in range(latency_matrix_size[1]):
#                 index = (i)*latency_matrix_size[1] + j
#                 x[index] = LpVariable("Cost (%d,%d)" % (i,j), 0, 1, LpInteger)
#         for l, i in zip(np.nditer(total_cost_matrix), range(total_cost_matrix.size)):
#             if i != 0:
#                 lp_objective += x[i]*l
#             else:
#                 lp_objective = x[i]*l
#         prob += lp_objective, "optimal latency decision"
#             # Contrainsts
#                 # Memory storage constraint
#         for j in range(1, latency_matrix_size[1]):
#             index = j
#             package_contraint = x[index]*package_size['package_size'][0]
#             for i in range(1, latency_matrix_size[0]):
#                 index = i*latency_matrix_size[1] + j
#                 package_contraint += x[index]* package_size['package_size'][i]
#             prob += package_contraint <= storageAvailable[j], "Sum of IoT packages must fit in storage M_%d" % j
#                 # Sum of device choices equals 1
#         for i in range(latency_matrix_size[0]):
#             index = i*latency_matrix_size[1]
#             sum_constraint = x[index]
#             for j in range(1, latency_matrix_size[1]):
#                 index = i*latency_matrix_size[1] + j
#                 sum_constraint += x[index]
#             prob += sum_constraint == 1, "Sum of decisions from IoT %d must equal 1" % i
#
#         #print(prob)
#         prob.solve()
#         print("Status:", LpStatus[prob.status])
#         if LpStatus[prob.status] != 'Optimal':
#             warnings.warn("Couldn't reach an optimal solution.")
#
#
#         decisions = [-1]*latency_matrix_size[0]
#         for v in prob.variables():
#             if v.varValue == 1:
#                 #parse v.name
#                 ij = v.name.split('(')[1].split(')')[0].split(',')
#                 decisions[int(ij[0])] = int(ij[1])
#
#         optimal_decisions.add(tuple(decisions))
#         if len(optimal_decisions) >= k:
#             break
#return optimal_decisions
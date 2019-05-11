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
from itertools import combinations, product
from operator import sub
from scipy.stats import norm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COST_TABLE_COLUMNS = ['cost', 'storage_available', 'k', 'optimal_decision']

# Define the state of our system at step k
# Attributes: current_storage_available (array) - storage available at each fog device,
#   k (int) - step in our system, cost (float) - The cost choosing optimal decisions from our state onwards
class State:
    def __init__(self, current_storage_available, k):#, previous_arrivals, current_storage_available, k):
        #self.previous_arrivals = previous_arrivals
        self.storage_available = current_storage_available
        self.k = k

# Calculates the distance between every IoT Device and every Fog Device into a matrix
# Input: IoT_devices (DataFrame) - set of IoT Devices, fog_devices (DataFrame) - set of fog devices
# Output: distanceMatrix (np 2d array) - matrix of distances from iot (rows) to fog (cols)
def distanceMatrix(df_IoT, df_fog):
    distanceMatrix = np.empty([df_IoT.shape[0], df_fog.shape[0]])

    for i in range(distanceMatrix.shape[0]):
        for j in range(distanceMatrix.shape[1]):
            distanceMatrix[i, j] = np.sqrt((df_fog.loc[j, 'xloc'] - df_IoT.loc[i, 'xloc']) ** 2 + (df_fog.loc[j, 'yloc'] + df_IoT.loc[i, 'yloc']) ** 2)
    return distanceMatrix

# Given the package sizes, calculate the possible set of memory thresholds that affect our decisions
# Input: storage_needed_per_package (list) - Storage memory required for the various different packages
# Output: memory_thresholds (set) - Memory thresholds in which we are interested
def memory_threshold_set():
    memory_thresholds = set()
    for r in range(1, len(PACKAGE_SIZES)+1):
        for comb in list(combinations(PACKAGE_SIZES, r)):
            thresh = sum(i for i in comb)
            if thresh <= STORAGE_MAX:
                memory_thresholds.add(thresh)
    memory_thresholds.add(0)
    return memory_thresholds


# We want to distribute the D IoT data packets to the F devices. Determine if a defined decision is valid
# Input: state (State) - the state of the system which includes the storage available at the fogs, decision (array) - size D array of decisions, send package d (IoT) to which fog id
# Output: valid.all() (boolean) - Whether the fogs have enough space to fit the packages, valid (array of boolean) - list of which values failed
def validDecision(state, decision):
    package = pd.DataFrame({'file_size': PACKAGE_SIZES})
    package = package.join(pd.DataFrame({'decision': decision}))
    package = package.groupby('decision')
    package = package.sum()
    fog_memory = pd.DataFrame({'storage_level': state.storage_available})
    memory_comparison = pd.merge(fog_memory, package,left_index=True, right_on='decision')
    valid = memory_comparison['storage_level'] >= memory_comparison['file_size']
    return valid.all()

# After validation, calculate the cost of a decision. There is no stochasticity involved in this cost since it is involved in the inputs
# Input: IoTDevices (DataFrame) - set of devices, fogDevices (DataFrame) - set of fog devices,
#   distance_matrix (np 2d array) - matrix of distances from IoT to fog, decision (array) - size D array of decision, send package d (IoT) to which fog id,
#   state (State) - the current state of the system, possible_state_storage (list(tuples) - The set of valid storage combinations
#   cost_table (DataFrame) - Current cost of each storage at each step
def decisionCostHat(df_IoT, df_fog, distance_matrix, decision,state, possible_state_storage, cost_table):
    package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size', 'upload_rate']])
    package = package.join(pd.DataFrame({'decision': decision}))
    package = pd.merge(package, pd.DataFrame(df_fog.loc[:, ['id', 'download_rate', 'storage_level']]), left_on='decision', right_on='id', suffixes=('_D', '_F'))
    package = package.sort_values(by='id_D').reset_index()
    package['latency'] = distance_matrix[package.loc[:, 'id_D'], package.loc[:, 'id_F']]
    #print(package.loc[:, ['file_size', 'upload_rate', 'download_rate', 'latency']])


    cost = package['file_size'].divide(package['upload_rate']).add(
        package['file_size'].divide(package['download_rate'])).add(
        2*package['latency']).sum()
    decision_storage_needed = package.loc[:, ['file_size', 'decision']]
    del package
    decision_storage_needed = decision_storage_needed.groupby('decision').sum()
    decision_storage_needed_list = [0]*df_fog.shape[0]
    for d, row in decision_storage_needed.iterrows():
        decision_storage_needed_list[d] = row['file_size']
    del decision_storage_needed
    # Sum of two independent normal distributions N(m1,v1) and N(m2, v2) is normal with N(m1+m2, v1+v2)
    normal_dist = df_fog.loc[:, ['storage_level', 'congestion_mean', 'congestion_var', 'processing_mean', 'processing_var']]
    normal_dist = normal_dist.join(pd.DataFrame({'storage_available': state.storage_available}))
    del df_fog

    # Below is the transition **********************************************************************************************
    normal_dist['mean'] = (normal_dist['processing_mean'] - normal_dist['congestion_mean']).multiply(
        normal_dist['storage_level']) - normal_dist['processing_mean'].multiply(normal_dist['storage_available'])
    normal_dist['var'] = (normal_dist['processing_mean'] - normal_dist['congestion_var']).multiply(
        normal_dist['storage_level']) - normal_dist['processing_var'].multiply(normal_dist['storage_available'])
    normal_dist.at[normal_dist['var'] < 1, 'var'] = 1

    storage_set = list(STORAGE_THRESHOLDS)
    fog_prob_states = []
    for idx, row in normal_dist.iterrows():
        if row['storage_level'] == np.inf:
            fog_prob = [1] * len(storage_set)
        else:
            fog_cdf = [0]*len(storage_set)
            fog_prob = [0] * len(storage_set)
            if row['var'] < 1:
                row['var'] = 1
            fog_norm = norm(loc=row['mean'], scale=row['var'])
            for i in range(len(storage_set)):
                fog_cdf[i] = fog_norm.cdf(storage_set[i])
            fog_prob[:-1] = [j-i for i, j in zip(fog_cdf[:-1], fog_cdf[1:])]
            prob = sum(fog_prob)
            fog_prob[-1] = 1-prob
        fog_prob_states.append(fog_prob)

    # Given the probabilities of being in each state, compute the Expected value of the next state cost
    df_state_storage = pd.DataFrame({'storage': possible_state_storage})
    df_state_storage.loc[:, 'prob'] = None
    for index, row in df_state_storage.iterrows():
        prob_storage = 1
        for idx, f in enumerate(row['storage'][1:]):
            prob_storage *= fog_prob_states[idx+1][list(STORAGE_THRESHOLDS).index(f)]
        df_state_storage.at[index, 'prob'] = prob_storage
    df_state_storage = df_state_storage[df_state_storage['prob'] != 0]
    df_state_storage.loc[:, 'prob'] *= 1./df_state_storage.sum()['prob'] # Scale to make up for lost percentages

    cost_table_next_state = cost_table[cost_table['k'] == state.k+1].copy()
    storage_tuple = [tuple(f) for f in cost_table.loc[:, 'storage_available'].tolist()]
    cost_table_next_state['storage_tuple'] = pd.DataFrame({'storage_tuple': storage_tuple})
    df_state_storage = pd.merge(df_state_storage, cost_table_next_state, left_on='storage', right_on='storage_tuple')
    # sum( prob(w)*cost(w) )
    df_state_storage['J_k+1'] = df_state_storage['prob'].multiply(df_state_storage['cost'])
    next_state_cost = df_state_storage.sum()['J_k+1']
        #next_state_cost = baseApproximationCalculate(state, distance_matrix, df_IoT, df_fog)

    return cost + next_state_cost


# Define all possible states given the fog descriptions
# Input: df_fog (DataFrame) - Description of all fog devices
# Output: possible_storage_levels (np 2darray) - All possible storage possibilities
def possibleStates(df_fog):
    storage_per_fog = []
    for index, row in df_fog.iterrows():
        fog = []
        if row['storage_level'] == np.inf:
            fog = [np.inf]
        else:
            for s in STORAGE_THRESHOLDS:
                if s <= row['storage_level']:
                    fog.append(s)
        storage_per_fog.append(fog)
    possible_storage_levels = [elem for elem in product(*storage_per_fog)]
    return possible_storage_levels

# Compare different expected costs and choose the minimum cost
def DynamicProgramming(df_IoT, df_fog, distance_matrix, N):
    print('Begin Dynamic Programming process')
    # Build the network of states for the last two layers only
    if N == 1:
        sys.exit("N must be greater than 1")
    # Add all possible states
    all_state_storage = possibleStates(df_fog)
    cost_table = pd.DataFrame(columns=COST_TABLE_COLUMNS)

    print('Number of States: %d' % len(all_state_storage))
    for state_storage in all_state_storage:
        init_state = State(list(state_storage), N)
        cost_table = cost_table.append(pd.DataFrame([[0, init_state.storage_available, init_state.k, None]], columns=COST_TABLE_COLUMNS), ignore_index=True)
    decision_set = set([decision for decision in product(list(range(df_fog.shape[0])), repeat=df_IoT.shape[0])])
    print("Decision size:" + str(len(decision_set)))

    for k in range(N-1, -1, -1):
        print('k=' + str(k) + ' at time: ' + str(time.time() - start))
        for state_storage in all_state_storage:
            state_k = State(list(state_storage), k)
            cost_table = cost_table.append(
                pd.DataFrame([[0, state_k.storage_available, state_k.k, None]], columns=COST_TABLE_COLUMNS), ignore_index=True)
        current_states = cost_table[cost_table.loc[:, 'k'] == k]
        for st_index, current_state in current_states.iterrows():
            cost_min = np.inf
            arg_min = None
            for decision in decision_set:
                #print('Checking decision: %s' % str(decision))
                if not validDecision(current_state, decision):
                    continue
                cost = decisionCostHat(df_IoT, df_fog, distance_matrix, decision, current_state, all_state_storage, cost_table)
                if cost < cost_min:
                    cost_min = cost
                    arg_min = decision
            cost_table.at[st_index, 'cost'] = cost_min
            cost_table.at[st_index, 'optimal_decision'] = arg_min

    cost_table.to_csv('optimal_solution.csv', index=False)


    return 0

# Main function of FogAssignment Problem
# Assign IoT tasks to Fog devices based on computation power, time, and congestion
def main(df_device, df_fog, N):
    dist_matrix = distanceMatrix(df_device, df_fog)
    DynamicProgramming(df_device, df_fog, dist_matrix, N=N)

def usage():
    print("-h Help\n-d csv file path to IoT device data\n-f csv file path to Fog device data\n-N Number of steps\n-s Random seed (optional)")

if __name__ == '__main__':
    # Start timer
    start = time.time()
    device_file = ''
    fog_file = ''
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
            device_file = arg
        elif opt == '-f':
            fog_file = arg
        elif opt == '-s':
            seed = int(arg)
        elif opt == '-N':
            N = int(arg)
        else:
            usage()
            sys.exit(2)
    if not device_file or not fog_file or N == 0:
        usage()
        sys.exit(2)

    np.random.seed(seed=seed)

    try:
        fogs = pd.read_csv(fog_file)
        devices = pd.read_csv(device_file)

    except:
        sys.exit("Was not able to read in Fog system with the given parameters")

    STORAGE_MAX = max(fogs.loc[1:, 'storage_level'].tolist())
    PACKAGE_SIZES = devices.loc[:, 'file_size'].tolist()
    STORAGE_THRESHOLDS = memory_threshold_set()

    main(devices, fogs, N)
    print("Time in sec: " + str(time.time() - start))

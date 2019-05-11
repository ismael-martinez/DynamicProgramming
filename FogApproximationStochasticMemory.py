# We are doing a simplified version of Fog Optimization for Dynamic Programming
# This script works with small instances for an exact solution.

import numpy as np
import scipy as sc
from tensorflow import keras, layers, nn
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("always")
import os
import random
import getopt
from operator import add
from itertools import combinations, product
from operator import sub
from scipy.stats import norm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COST_TABLE_COLUMNS = ['cost', 'storage_available', 'k', 'optimal_decision']
P = 3

# Define the state of our system at step k
# Attributes: current_storage_available (array) - storage available at each fog device,
#   k (int) - step in our system, cost (float) - The cost choosing optimal decisions from our state onwards
class State:
    def __init__(self, current_storage_available, k):#, previous_arrivals, current_storage_available, k):
        #self.previous_arrivals = previous_arrivals
        self.storage_available = current_storage_available
        self.k = k

def storage_k_oh(series_lr):
    k = series_lr['k']
    series_lr = series_lr.drop('k')
    indices = series_lr.index
    new_indices = []
    for i in indices:
        new_indices.append(i + '_k')
    new_series = pd.Series(index=new_indices)
    for idx, val in series_lr.iteritems():
        new_idx = idx + '_k'
        if series_lr[idx] == 1:
            new_series[new_idx] = k
        else:
            new_series[new_idx] = 0
    return new_series


# Read in optimal solution for Exact Solution, and build a model to approximate the costs
# Input: df_optimal_solution (DataFrame) - Data from solution of exact solution
# Ouptut: df_optimal_solution_oh_col (DataFrame desc) - One hot encoded stage variables column labels only, model (sm.regression.linear_model) - LR model to estimate cost
def baseApproximationTrainingModel(df_optimal_solution):
    # Parse
    df_optimal_solution = df_optimal_solution.dropna() # Gets rid of the 0th cost as well
    target = df_optimal_solution.loc[:, 'cost'].values
    df_optimal_solution = df_optimal_solution.drop(['cost', 'optimal_decision'], axis=1).reset_index(drop=True)
    N_prime = df_optimal_solution.loc[df_optimal_solution['k'].idxmax()].k
    df_optimal_solution['k'] = df_optimal_solution['k'] + N - N_prime

    # Expand k to polynomial
    df_optimal_solution['const'] = 1
    for p in range(2, P+1):
        col = 'k_%d' % p
        df_optimal_solution[col] = df_optimal_solution['k']**p
    storage = df_optimal_solution['storage_available']
    storage = pd.get_dummies(storage)
    storage_k = storage.join(df_optimal_solution['k'])
    storage_k = storage_k.apply(storage_k_oh, axis=1)
    df_optimal_solution_oh = df_optimal_solution.join(storage)
    df_optimal_solution_oh = df_optimal_solution_oh.join(storage_k)
    df_optimal_solution_oh = df_optimal_solution_oh.drop(columns='storage_available')

    inputs_lr = df_optimal_solution_oh.values

    # Linear Regression without normalization on k
    lr_model = sm.OLS(target, inputs_lr).fit()
    # Print out the statistics
    print(lr_model.summary())
    prediction = lr_model.predict(inputs_lr)  # make the predictions by the model
    err = [np.abs((j-i)/(j+1)) for i , j in zip(prediction, target)]
    print('Mean error: ' + str(np.mean(err)))

    return [df_optimal_solution_oh.columns, lr_model]

# Given a state, determine the base approximation cost
# Input: state (Series) - the state for who's base approximation we intend to find
# Output: ba_cost (float) - Calculated cost
def baseApproximationCalculate(state):
    if state.k == N:
        return 0
    # if state.k >= 5:
    #     print(state)
    #     pass
    data = [0]*df_optim_sln_oh_columns.shape[0]
    df_input = pd.DataFrame([data], columns=df_optim_sln_oh_columns)
    df_input.at[0, 'k'] = state['k']
    df_input.at[0, 'const'] = 1
    for p in range(2, P + 1):
        col = 'k_%d' % p
        df_input.at[0, col] = state['k'] ** p
    storage = state['storage']
    storage = list(storage)
    storage_k = str(storage) + '_k'
    df_input.at[0, str(storage)] = 1
    df_input.at[0, storage_k] = state['k']
    input_lr = df_input.values
    prediction = lr_model.predict(input_lr)[0]  # make the predictions by the model

    return prediction

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
def decisionCostHat(df_IoT, df_fog, distance_matrix, decision, state, possible_state_storage):
    if state.k == N:
        return 0
    package = pd.DataFrame(df_IoT.loc[:, ['id', 'file_size', 'upload_rate']])
    package = package.join(pd.DataFrame({'decision': decision}))
    package = pd.merge(package, pd.DataFrame(df_fog.loc[:, ['id', 'download_rate', 'storage_level']]), left_on='decision', right_on='id', suffixes=('_D', '_F'))
    package = package.sort_values(by='id_D').reset_index()
    package['latency'] = distance_matrix[package.loc[:, 'id_D'], package.loc[:, 'id_F']]
    #print(package.loc[:, ['file_size', 'upload_rate', 'download_rate', 'latency']])

    # Cost of the decision from this state to the next
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
            fog_cdf = [0] * len(storage_set)
            fog_prob = [0] * len(storage_set)
            if row['var'] < 1:
                row['var'] = 1
            fog_norm = norm(loc=row['mean'], scale=row['var'])
            for i in range(len(storage_set)):
                fog_cdf[i] = fog_norm.cdf(storage_set[i])
            fog_prob[:-1] = [j - i for i, j in zip(fog_cdf[:-1], fog_cdf[1:])]
            prob = sum(fog_prob)
            fog_prob[-1] = 1 - prob
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
    df_state_storage['k'] = state.k+1
    df_state_storage['apprCost'] = None

    df_state_storage['apprCost'] = df_state_storage.apply(baseApproximationCalculate, axis=1)
    df_state_storage['expectedCost'] = df_state_storage['prob'].multiply(df_state_storage['apprCost'])
    # for index, row in df_state_storage.iterrows():
    #  df_state_storage.at[index, 'apprCost'] = baseApproximationCalculate(row)
    # df_state_storage['expectedCost'] = df_state_storage['apprCost'].multiply(df_state_storage['prob'])
    # next_state_cost = df_state_storage['expectedCost'].sum()
    next_state_cost = df_state_storage['expectedCost'].sum()

    return cost + max(next_state_cost, 0) # Can't have below 0 cost

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

# Do an approximative forward pass using the baseApproximation
def DynamicProgramming(df_IoT, df_fog, distance_matrix, N):
    cost_table = pd.DataFrame([], columns=COST_TABLE_COLUMNS)
    # Determine the set of possible states for step k
    possible_states = possibleStates(df_fog)
    # Choose optimal policy by taking the min cost incurred by different decisions
    decision_set = set([decision for decision in product(list(range(df_fog.shape[0])), repeat=df_IoT.shape[0])])
    for k in range(N+1):
        print('k=' + str(k) + ' at time: ' + str(time.time() - start))
        for state_storage in possible_states:
            state_k = State(list(state_storage), k)
            cost_table = cost_table.append(
                pd.DataFrame([[0, state_k.storage_available, state_k.k, None]], columns=COST_TABLE_COLUMNS),
                ignore_index=True)
        current_states = cost_table[cost_table.loc[:, 'k'] == k]
        for st_index, current_state in current_states.iterrows():
            cost_min = np.inf
            arg_min = None
            for decision in decision_set:
                # print('Checking decision: %s' % str(decision))
                if not validDecision(current_state, decision):
                    continue
                cost = decisionCostHat(df_IoT, df_fog, distance_matrix, decision, current_state, possible_states)
                if cost < cost_min:
                    cost_min = cost
                    arg_min = decision
            cost_table.at[st_index, 'cost'] = cost_min
            cost_table.at[st_index, 'optimal_decision'] = arg_min
    cost_table.to_csv('optimal_solution_approximation.csv')
    return 0

# Main function of FogAssignment Problem
# Assign IoT tasks to Fog devices based on computation power, time, and congestion
def main():
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
    optim_sln = ''
    seed = None
    debug = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:f:s:N:o:")
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
        elif opt == '-o':
            optim_sln = arg
        else:
            usage()
            sys.exit(2)
    if not device_file or not fog_file or N == 0 or not optim_sln :
        usage()
        sys.exit(2)

    np.random.seed(seed=seed)

    try:
        df_fog = pd.read_csv(fog_file)
        df_device = pd.read_csv(device_file)
        df_optim_sln = pd.read_csv(optim_sln)
        df_optim_sln_oh_columns, lr_model = baseApproximationTrainingModel(df_optim_sln)

    except:
        sys.exit("Was not able to read in Fog system with the given parameters")

    STORAGE_MAX = max(df_fog.loc[1:, 'storage_level'].tolist())
    PACKAGE_SIZES = df_device.loc[:, 'file_size'].tolist()
    STORAGE_THRESHOLDS = memory_threshold_set()

    main()
    print("Time in sec: " + str(time.time() - start))

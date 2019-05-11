import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt
import time

def main():
    print(df_optimal_approx.head())
    print(df_optimal_exact.head())
    # Compare costs
    states = df_optimal_exact['storage_available']
    states = set(states)
    exact = df_optimal_exact[df_optimal_exact['storage_available'] =='[inf, 1, 1]']
    #exact = exact[exact['k'] != 4]
    N_e = df_optimal_exact.loc[df_optimal_exact['k'].idxmax()].k
    approx = df_optimal_approx[df_optimal_approx['storage_available'] == '[inf, 1, 1]']
    #approx = approx[approx['k'] != 10]
    N_a = df_optimal_approx.loc[df_optimal_approx['k'].idxmax()].k

    print(exact)
    print(approx)
    x_e = exact['k'].tolist()
    y_e = exact['cost'].tolist()
    plt.scatter(x_e, y_e, label='Exact')

    x_a = approx['k'].tolist()
    y_a = approx['cost'].tolist()
    plt.scatter(x_a, y_a, label='Approximate')

    plt.xlabel('k')
    plt.ylabel('Cost')
    plt.xticks(range(max(N_e+1, N_a)))
    plt.legend(loc='lower left')
    plt.show()


    return 0

def usage():
    print("-h Help\n-e <path to optimal solution set of exact solution> \n-a <path to optimal solution of approximate solution")

if __name__ == '__main__':
    # Start timer
    start = time.time()
    exact_optimal = ''
    approx_optimal = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:a:")
    except getopt.GetoptError as err:
        usage()
        sys.exit('The command line inputs were not given properly')
    for opt, arg in opts:
        if opt == '-e':
            exact_optimal = arg
        elif opt == '-a':
            approx_optimal = arg
        else:
            usage()
            sys.exit(2)
    if not approx_optimal or not exact_optimal:
        usage()
        sys.exit(2)

    try:
        df_optimal_exact = pd.read_csv(exact_optimal)
        df_optimal_approx = pd.read_csv(approx_optimal)

    except:
        sys.exit("Was not able to read in Fog system with the given parameters")
    main()
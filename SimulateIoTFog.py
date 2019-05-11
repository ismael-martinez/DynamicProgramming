import numpy as np
import time
import sys
import getopt
import scipy as sc
import pandas as pd
giga_to_mega = 1024 # 1024 MB = 1 GB

# A device IoT node D has fixed memory and base time. r~N(c*m)
# Input: memory (int) - megabytes per data point
class IoT_Device:
    c = 5
    def __init__(self, memory):
        self.memory = memory
        self.base_time = np.random.normal(self.c*memory, 1)

# A fog device F has fixed processing power and a memory cap, M~N(s*rho)
# Memory cap in Giga Bytes
# Input: processing_power (int) - megabytes processed per second
class Fog_Device:
    s_lat = 0.5
    s_mem = 0.5
    def __init__(self, processing_power, cloud=False):
        self.processing_power = processing_power
        self.latency = np.random.normal(self.s_lat * np.exp(self.processing_power), 1)
        if cloud:
            self.memory_cap = np.inf
        else:
            self.memory_cap = np.random.normal(self.s_mem * self.processing_power, 1)

# Given the number of IoT Devices and Fog Devices, create a system with all parameters inplace
def main(device_num, fog_num, seed):
    np.random.seed(seed) # Set seed for the whole process

    min_mem = 0.5 # MB
    max_mem = 5
    D = []
    for d in range(device_num):
        if d == 0:
            D.append(IoT_Device(min_mem))
        elif d == 1:
            D.append(IoT_Device(max_mem))
        else:
            D.append(IoT_Device(np.random.rand()*(max_mem-min_mem)+min_mem))

    for d in D:
        print(d.memory)
        print(d.base_time)
        print("\n")

    min_memory_cap = 0.5  # 500 MB
    max_memory_cap = 32
    F = []
    for f in range(fog_num):
        if f == 0:
            F.append(Fog_Device(max_memory_cap*2, cloud=True))
        elif f == 1:
            F.append(Fog_Device(max_memory_cap))
        elif f == 2:
            F.append(Fog_Device(min_memory_cap))
        else:
            F.append(Fog_Device(np.random.rand()*(max_memory_cap-min_memory_cap)+min_memory_cap))
    for f in F:
        print(f.memory_cap)
        print(f.processing_power)
        print(f.latency)
        print("\n")


def usage():
    print("-h Help\n-d Number of IoT devices\n-f Number of Fog devices\n-s Random seed (optional)")

if __name__ == '__main__':
    # Start timer
    start = time.time()
    device_num = 0
    fog_num = 0
    seed = None
    debug = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:f:s:")
    except getopt.GetoptError as err:
        usage()
        sys.exit('The command line inputs were not given properly')
    for opt, arg in opts:
        if opt == '-d':
            device_num = int(arg)
        elif opt == '-f':
            fog_num = int(arg)
        elif opt == '-s':
            seed = arg
        else:
            usage()
            sys.exit(2)
    if device_num == 0 or fog_num == 0:
        usage()
        sys.exit(2)

    main(device_num, fog_num, seed)
    print("Time in sec: " + str(time.time() - start))

# We are working with a simplified version of Fog Latency optimization,
# This script is for simulating the fog architecture given a D (number of IoT Devices), F (number of fog) and seed (random seed for predictable instance)

import numpy as np
import scipy as sc
import pandas as pd
import time
import sys
import os
import getopt

RADIUS = 0.5
SPEED_OF_INTERNET = 150000
TRANSFER_RATES = [5.5, 11, 20, 24, 54, 100, 200, 600, 1300]
IOT_UPLOAD_RATES = [5.5, 11, 20, 24]
DISTANCES = [1, 2, 5, 7, 10]#[1,2,5,10,15,20,50,75,100,250,375,500,1000,2000,5000]
STORAGE_LIMIT = 100#[4,8,16,32,64,128,256,512,1024,2048]
STORAGE_MIN = 25
#POISSON_ARRIVAL_RATE = [1, 2, 3]
giga_to_mega = 1024 # 1024 MB = 1 GB

#Given a distance from 0,0, calculate a x,y pair that has this distance
# Input: distance (int) - Int between 0.5 and 20,000
def location(distance):
    x = np.random.rand()*distance
    y = np.sqrt(distance**2 - x**2)
    return [x,y]

# FogDevice class, containing computation cap and resourcesLeft at any given time
class FogDevice:
    def __init__(self, storage_level, congestion_mean, congestion_var, processed_mean, processed_var, distance, download_rate):
        # self.storageCap = storageCap * giga_to_mega
        # self.storageLeft = storageCap * giga_to_mega
        self.storage_level = storage_level
        self.congestion_mean =  congestion_mean
        self.congestion_var = congestion_var
        self.processed_mean = processed_mean
        self.processed_var = processed_var
        self.distance = distance
        self.location = location(distance)
        self.download_rate = download_rate


# Input: fog (class FogDevice) - device from which to return attributes in df order
def fogAttributes(fog):
    return [fog.storage_level, fog.congestion_mean, fog.congestion_var, fog.processed_mean, fog.processed_var, fog.distance, fog.location[0], fog.location[1], fog.download_rate]


# We simulate K fog devices, with a lower bound and an upper bound on the computation power of each.
# At least one device must satisfy the upper bound and one the lower bound
# Input: K (int) - Number of fog devices, min (float) - Minimum computation power, max (float) - Maximum computation power
# Output: Devices (array) - Collection of fog devices
def fogDevices(K):
    if K < 2:
        sys.exit("The number of Fog Devices must be at least 2.")
    columns = ['storage_level', 'congestion_mean', 'congestion_var', 'processing_mean', 'processing_var', 'distance', 'xloc', 'yloc', 'download_rate']
    df_fog = pd.DataFrame(index=range(K), columns=columns)
    fog = FogDevice(np.inf, 1, 0, 1, 0, 100, 1300)
    df_fog.loc[0,:] = fogAttributes(fog) # CLOUD
    fog = FogDevice(5, 0.2, 0.1, 0.2, 0.1, 0.5, TRANSFER_RATES[0])
    df_fog.loc[K-1, :] = fogAttributes(fog)

    distances = []
    rates = []
    storage = []
    congestion_mean = []
    congestion_var = []
    processed_mean = []
    processed_var = []
    for i in range(2, K):
        distances.append(np.random.choice(DISTANCES))
        rates.append(np.random.choice(TRANSFER_RATES))
        storage.append(np.random.randint(STORAGE_MIN, STORAGE_LIMIT))
        congestion_mean.append(np.random.random())
        congestion_var.append(np.random.random()*0.1)
        processed_mean.append(np.random.random())
        processed_var.append(np.random.random()*0.1)
    distances.sort(reverse=True)
    rates.sort(reverse=True)
    storage.sort(reverse=True)
    for i in range(1, K-1):
        fog = FogDevice(storage[i-1], congestion_mean[i-1], congestion_var[i-1], processed_mean[i-1], processed_var[i-1],
                        distances[i-1], rates[i-1])
        df_fog.loc[i,:] = fogAttributes(fog)
    return df_fog

# IoTDevice class, containing IoT id, computation needs, gamma (mean time required), start and compTime to be defined later
class IoTDevice:
    def __init__(self, file_size, upload_rate):
        self.file_size = file_size
        distance = np.random.rand()*RADIUS
        self.location = location(distance)
        self.upload_rate = upload_rate
        #self.poisson_arrival_rate = random.choice(POISSON_ARRIVAL_RATE)

# Input: IoT (class IoTDevice) - Device from which to take attributes for DataFrame
def IoTAttributes(IoT):
    return [IoT.file_size, IoT.location[0], IoT.location[1], IoT.upload_rate]#, IoT.poisson_arrival_rate]

def IoTDevices(K):
    if K < 1:
        sys.exit("The number of IoT Devices must be at least 1.")
    columns = ['file_size', 'xloc', 'yloc', 'upload_rate']#, 'poisson_arrival_rate']
    df_IoT = pd.DataFrame(index=range(K), columns=columns)
    for i in range(K):
        iot = IoTDevice(np.random.randint(1, 3), np.random.choice(IOT_UPLOAD_RATES))
        df_IoT.loc[i,:] = IoTAttributes(iot)
    return df_IoT

# Main function of FogAssignment Problem
# Assign IoT tasks to Fog devices based on computation power, time, and congestion
def main(D, F, sd):
    np.random.seed(seed=sd)
    try:
        fogs = fogDevices(F)
        fogFile = 'fogData_seed.csv'
        fogs.to_csv(fogFile, index_label='id')

        devices = IoTDevices(D)
        deviceFile = 'deviceData_seed.csv'
        devices.to_csv(deviceFile, index_label='id')
    except:
        sys.exit("Was not able to create Fog system with the given parameters")


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
            seed = int(arg)
        else:
            usage()
            sys.exit(2)
    if device_num == 0 or fog_num == 0:
        usage()
        sys.exit(2)

    main(device_num, fog_num, seed)
    print("Time in sec: " + str(time.time() - start))

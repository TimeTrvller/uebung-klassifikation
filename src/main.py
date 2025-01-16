"""
Python code für Struktur-& Objektextraktion in 2D & 3D
Übung 2: semantische Klassifizierung
"""
# import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

filepath = './data/'
filename = 'point_cloud_data.mat'

with h5py.File(filepath+filename,'r') as file:
    # data structure:
    # ['PC_training', 'PC_validation'] je x,y,z,class als Reihe

    train_data = file['PC_training']
    valid_data = file['PC_validation']
    print(train_data.shape)
    print(type(train_data))

    xt = train_data[0,:]
    yt = train_data[1,:]
    zt = train_data[2,:]
    klt = train_data[3,:]

    xv = valid_data[0,:]
    yv = valid_data[1,:]
    zv = valid_data[2,:]
    klv = valid_data[3,:]

####################

xyzt = np.column_stack((xt,yt,zt))
xyzv = np.column_stack((xv,yv,zv))

print(xyzt.shape)
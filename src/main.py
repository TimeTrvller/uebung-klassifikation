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

file = h5py.File(filepath+filename,'r')

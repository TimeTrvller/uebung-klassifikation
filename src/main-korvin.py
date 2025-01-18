import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import h5py


#? --- AUFGABE 1 -------------------------------------------------------------
"""
Laden Sie die gegebene Datei 'point_cloud_data.mat' in Ihrem Python-Skript (z.B. mit dem Modul h5py).
Diese Datei enthält zwei (n x 4)-Matrizen, welche für n 3D-Punkte jeweils die XYZ-Koordinaten sowie
die Klassenzugehörigkeit enthalten. Erstellen Sie eine Funktion, die für eine (n x 3)-Matrix
mit XYZ-Koordinaten für n Punkte als Eingangsgröße einen (n x k)- Vektor der in den jeweiligen
Nachbarschaften enthaltenen Punkte angibt.
"""                
                   
filepath = './data/'
filename = 'point_cloud_data.mat'

with h5py.File(filepath+filename,'r') as file:
    # extract the training and validation data
    train_data = file['PC_training']    # 4xn matrix (x,y,z,class)
    print(train_data.shape)
    valid_data = file['PC_validation']
    
    points3d_train = train_data[:3,:].T
    points3d_valid = valid_data[:3,:].T


def getNeighborhood(points, k):
    """
    Compute the k nearest neighbors for each point of points.
    
    Parameters:
      points:   (n x 3)-Matrix with n 3D points (x,y,z)
        k   :   number of neighbors (int)
    """
    
    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)

    # Find the k+1 nearest neighbors (including the point itself)
    distances, indices = nbrs.kneighbors(points)

    # Exclude the first neighbor (itself) if needed
    indices = indices[:, 1:]

    return indices
    
    
    
#? --- AUFGABE 2 -------------------------------------------------------------
"""
Erstellen Sie eine Funktion, mit der für eine (n x 3)-Matrix mit XYZ-Koordinaten
die entsprechenden Covariance Features berechnet werden.
Hinweis: Die Funktion soll eine (n x 8)-Matrix liefern.
"""
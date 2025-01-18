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

# Load the data             
                   
filepath = './data/'
filename = 'point_cloud_data.mat'

with h5py.File(filepath+filename,'r') as file:
    # extract the training and validation data
    train_data = file['PC_training']    # 4x36932 matrix (x,y,z,class)
    valid_data = file['PC_validation']  # 4x91515 matrix (x,y,z,class)
    
    points3d_train = train_data[:3,:].T
    points3d_valid = valid_data[:3,:].T


def getNeighborhood(points, k, firstNeighbor=False):
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
    if firstNeighbor:
        indices = indices[:, 1:]
        
    # Get the coordinates of the neighbors
    points_neighbors = points[indices]

    return indices, points_neighbors

# Test the function
points = points3d_train
k = 50
indices_neighbors, points_neighbors = getNeighborhood(points, k)

    
#? --- AUFGABE 2 -------------------------------------------------------------
"""
Erstellen Sie eine Funktion, mit der für eine (n x 3)-Matrix mit XYZ-Koordinaten
die entsprechenden Covariance Features berechnet werden.
Hinweis: Die Funktion soll eine (n x 8)-Matrix liefern.
"""


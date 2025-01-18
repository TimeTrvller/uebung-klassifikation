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
    # Extract the training and validation data
    train_data = file['PC_training']    # 4x36932 matrix (x,y,z,class)
    valid_data = file['PC_validation']  # 4x91515 matrix (x,y,z,class)
    
    points3d_train = train_data[:3,:].T # 36932x3 matrix (x,y,z)
    points3d_valid = valid_data[:3,:].T # 91515x3 matrix (x,y,z)


def getNeighborhood(points: np.ndarray, k: int, firstNeighbor: bool = False):
    """
    Compute the k nearest neighbors for each point of points.
    
    Parameters:
        points            : (n x 3)-Matrix with n 3D points (x,y,z)
        k                 : number of neighbors
        firstNeighbor     : if True, exclude the point itself from the neighbors
    Returns:
        indices_neighbors : (n x k)-Matrix with the indices of the neighbors
        points_neighbors  : (n x k x 3)-Matrix with the coordinates of the neighbors
    
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
points_train = points3d_train
k = 50
indices_neighbors_train, points_neighbors_train = getNeighborhood(points_train, k, firstNeighbor=True)
    
#? --- AUFGABE 2 -------------------------------------------------------------
"""
Erstellen Sie eine Funktion, mit der für eine (n x 3)-Matrix mit XYZ-Koordinaten
die entsprechenden Covariance Features berechnet werden.
Hinweis: Die Funktion soll eine (n x 8)-Matrix liefern.
"""

def getCovFeatures(points_neighbors: np.ndarray):
    """
    Compute the covariance features for each point of points_neighbors.
    Covariance features are the 8 entries of the scattermatrix.
    
    linearity, planarity, scattering, omnivariance, anisotropy,
    eigenentropy, sum of eigenvalues, change of curvature
    
    Parameters:
        points_neighbors : (n x k x 3)-Matrix with n 3D points and their k neighbors (x,y,z)
    Returns:
        cov_features     : (n x 8)-Matrix with the covariance features
    """
    
    # Get the number of points
    n = points_neighbors.shape[0]
    
    # Initialize the matrix to store the covariance features
    cov_features = np.zeros((n, 8))
    
    # Compute the covariance features for each point
    for i in range(n):
        # Get the neighbors coordinates (x,y,z) of the point i
        neighbors = points_neighbors[i]
        
        #! Compute the covariance matrix
        num_neighbors = neighbors.shape[0]           # number of neighbors
        sample_mean   = np.mean(neighbors, axis=0)   # mean of the neighbors
        cov_matrix    = 1/(num_neighbors - 1) * (neighbors - sample_mean).T @ (neighbors - sample_mean) # formula from the script
        
        #! Compute the eigenvalues
        eigenvalues, _ = np.linalg.eigh(cov_matrix)   # eigenvectors are not needed
        eigenvalues    = np.flip(eigenvalues)         # order largest first
        
        #! Compute the covariance features
        lbd1, lbd2, lbd3 = eigenvalues
        
        linearity    = (lbd1 - lbd2) / lbd1
        planarity    = (lbd2 - lbd3) / lbd1
        scattering   = lbd3 / lbd1
        omnivariance = np.cbrt(lbd1 * lbd2 * lbd3)
        anisotropy   = (lbd1 - lbd3) / lbd1
        eigenentropy = - sum(lbd * np.log(lbd) for lbd in (lbd1, lbd2, lbd3))
        sum_eigen    = sum(eigenvalues)
        change_curv  = lbd3 / sum_eigen
        
        # Store the covariance features
        cov_features[i] = [linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, sum_eigen, change_curv]
    
    return cov_features
    
# Test the function
cov_features_train = getCovFeatures(points_neighbors_train)


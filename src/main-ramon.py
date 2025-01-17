"""
Python code für Struktur-& Objektextraktion in 2D & 3D
Übung 2: semantische Klassifizierung
"""
# import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import h5py


# function to get neighbourhood:
def getNeighborhood(inputXYZ, numNeighbors):
    """
    Compute the k-neighborhood of every point of a input point cloud.
    Returns for every point of input point cloud the indices of the neighbors.
    Variables:  inputXYZ:       input point cloud
                numNeighbors:   number of neighbors to be computed
    Returns:    indices:        indices of neighbors in point cloud
                neighborsXYZ:   3D-np-array with k neighbors of each point
                                dim-0: points
                                dim-1: k neighbors
                                dim-2: xyz value
    """
    lenXYZ = np.shape(inputXYZ)[0]
    dimension = np.shape(inputXYZ)[1]
    neighborsXYZ = np.zeros([lenXYZ, numNeighbors, dimension])

    # scikit-learn implementation to get indices
    nbrs = NearestNeighbors(n_neighbors=(numNeighbors+1), algorithm='ball_tree').fit(inputXYZ)
    distances , indices = nbrs.kneighbors(inputXYZ)
    indices = np.delete(indices, (0), axis=1)       # delete own point

    # get positions of points(idices)
    for i in np.arange(lenXYZ):
        for j in np.arange(numNeighbors):
            neighborsXYZ[i,j,:] = inputXYZ[indices[i,j],:]

        # visualize each point and his nearest neighbors
        if i == 1:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(inputXYZ[:,0],inputXYZ[:,1],inputXYZ[:,2],marker='.')
            ax.scatter(inputXYZ[i,0],inputXYZ[i,1],inputXYZ[i,2],marker='o', c='red', label='input point')
            ax.scatter(neighborsXYZ[i,:,0],neighborsXYZ[i,:,1], neighborsXYZ[i,:,2],marker='o',c='green', label="2 nearest neighbors")
            plt.legend()
            plt.show()
    return indices, neighborsXYZ


# main program
filepath = '../data/'       # Linux
# filepath = './data/'      # Windows
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



# Testing point cloud
testCloud = np.random.rand(10,3)
numNeighbors = 5

idx, neighborsXYZ = getNeighborhood(testCloud, numNeighbors)
print('neighborsXYZ', neighborsXYZ)
print('idx',idx)

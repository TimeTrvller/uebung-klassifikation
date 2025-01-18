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
                neighborsXYZ:   3D-np-array wit*h k neighbors of each point
                                dim-0: points
                                dim-1: k neighbors
                                dim-2: xyz value
    """
    lenXYZ = np.shape(inputXYZ)[0]
    dimXYZ = np.shape(inputXYZ)[1]
    neighborsXYZ = np.zeros([lenXYZ, numNeighbors, dimXYZ])

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
            ax.scatter(neighborsXYZ[i,:,0],neighborsXYZ[i,:,1], neighborsXYZ[i,:,2],marker='o',c='green', label="nearest neighbors")
            plt.legend()
            plt.show()
    return indices, neighborsXYZ

def getCovMatrix(inputXYZ):
    """
    Compute the covariance matrix of a (sample) point cloud inputXYZ. This is also called the 3D structure tensor.
    Link: https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
    """
    numPoints = np.size(inputXYZ,0)
    sampleMean = np.mean(inputXYZ,0)     # compute the mean of all points (dim=0)

    inputXYZNormed = inputXYZ - sampleMean                      # transfor to weighted coordinates
    scatterMatrix = inputXYZNormed.T @ inputXYZNormed           # the outer matrix product does the work of the sum in the formula (c.f. readme.md)
    #scatterMatrix = np.dot(inputXYZNormed.T, inputXYZNormed)   # alternative code for the outer product

    covMatrix = 1 / (numPoints - 1) * scatterMatrix             # Error correction frm script: (numPoints-1) https://en.wikipedia.org/wiki/Sample_mean_and_covariance
    return covMatrix

def getEigenValues(covMatrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covMatrix)
    eigenvalues = np.flip(eigenvalues)                      # order largest first
    
    print("eigenvalues", eigenvalues)
    
    

    lambda1 = eigenvalues[0]                                # largest eigenvalue
    lambda2 = eigenvalues[1]
    lambda3 = eigenvalues[2]                                # smallest eigenvalue

    return lambda1, lambda2, lambda3

def getCovarianceFeatures(inputXYZ, numNeighbors):
    """
    Compute the covariance features for each point of inputXYZ.
    Covariance features are the 8 entries of the scattermatrix
    """
    neighborsXYZ = getNeighborhood(inputXYZ,numNeighbors)

    # todo: getScatterMatrix() for each point
    #
    # todo: do PCA for each scattermatrix
    #
    # todo: get eigenvalues from PCA
    #
    # todo: compute the eight features for each point
    #
    # do all this in a loop by calling functions
    #return covarianceFeatures
    pass


# main program
# filepath = '../data/'       # Linux
filepath = './data/'      # Windows
filename = 'point_cloud_data.mat'

with h5py.File(filepath+filename,'r') as file:
    # data structure:
    # ['PC_training', 'PC_validation'] je x,y,z,class als Reihe

    train_data = file['PC_training']
    valid_data = file['PC_validation']
    #print(train_data.shape)
    #print(type(train_data))

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
#print('neighborsXYZ', neighborsXYZ)
#print('idx',idx)

covMatrix = getCovMatrix(testCloud)
print("covMatrix", covMatrix)
print("eigenValues", getEigenValues(covMatrix))

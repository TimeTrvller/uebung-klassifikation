# Semantic classification of 3d point clouds

The goal of this exercise is the implementation of a procedure for semantic classification of a 3d point cloud with a random forest classificator.
We chose a neighborhood of k=50 nearest neighbors.<br>
The exercise consists of the following steps.

## Neighborhood
The input data `point_cloud_data.mat` are loaded with the module `h5py`. They consist of two nx4 matrices.
Each matrix consists of XYZ coordinates and the point class. The first step is to implement a function to get the (nx3) matrix of neighbors for each point of a point cloud.
This function is implemented as `getNeighborhood()`.

## Covariance features
For each point, a total of eight features based on the eigenvalues of the *scatter matrix* are computed. This is done in the functions `getScatterMatrix()` and `computeCovarianceFeatures()`.
The *scatter matrix* or *structure tensor* is given by the 3D covariance matrix
```math
S3D = \frac{1}{k-1} \cdot \sum_{i=0}^{k} (X_i - \bar{X})(X_i - \bar{X})^T
```
Once the eigenvalues
```math
\lambda_1 \geq \lambda_2 \geq \lambda3 \geq 0
```
are computed by PCA, they are used to compute the following eight features:

### Linearity
```math
L_\lambda = \frac{\lambda_1 - \lambda_2}{\lambda_1}
```

### Planarity
```math
P_\lambda = \frac{\lambda_2 - \lambda_3}{\lambda_1}
```

### Scattering
```math
S_\lambda = \frac{\lambda_3}{\lambda_1}
```

### Omnivariance
```math
O_\lambda = \sqrt[\leftroot{10} \uproot{5} 3]{\lambda_1\cdot\lambda_2\cdot\lambda_3}
```

### Anisotropy
```math
A_\lambda = \frac{\lambda_1 - \lambda_3}{\lambda_1}
```

### Eigentropy
```math
A_\lambda = -\sum_{i=1}^{3} \lambda_i \ln{\lambda_i}
```

### Sum of eigenvalues
```math
\Sigma_\lambda = \lambda_1 + \lambda_2 + \lambda_3
```

### Change of curvature
```math
C_\lambda = \frac{\lambda_3}{\lambda_1+\lambda_2+\lambda_3}
```

## Classification
After the features are computed for each point of the point cloud, the classification with random forest classificatior is executed.
The classificator is trained with the dataset `PC_training` contained in `\data\point_cloud_data.mat` and validated on the dataset `PC_validation`.

## Evaluation
The quality of the classification is determined by computing the confusion matrix and other quality measures.

## Visualisation
After classification, the colored point cloud is exported as `.ply` with `helper_functions.py` and visualized in `Meshlab`.

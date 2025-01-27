# Semantic classification of 3d point clouds

The goal of this exercise is the implementation of a procedure for semantic classification of a 3d point cloud with a random forest classificator.
We chose a neighborhood of $k=50$ nearest neighbors.<br>
The exercise consists of the following steps.

## Neighborhood
The input data `point_cloud_data.mat` are loaded with the module `h5py`. They consist of two nx4 matrices.
Each matrix consists of XYZ coordinates and the point class. The first step is to implement a function to get the (nx3) matrix of neighbors for each point of a point cloud.
This function is implemented as `getNeighborhood()`.

## Covariance features
For each point, a total of eight features based on the eigenvalues of the *scatter matrix* are computed. This is done in the function `getCovFeatures()`.
The *structure tensor* is given by the 3D covariance matrix:
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
The quality of the classification is determined by computing the confusion matrix and other quality measures, which rely on the Number of true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN).
The first four metrics are computed for each class $i$. The last two metrics give information about the overall classification in one number. In the following, there are index $i$ runs from $1$ to the number of Classes $N$.

### Recall
The $Recall$ of class $i$ is the proportion between correctly classified actual positive instances of the class and all actual positives for this class. This metric is therefore also called *Completeness*, because it is a measure of how well/complete the positive instances are actually correctly classified. The Recall is also equal to the Probability, that a Point classified as class A is actually in class A.
```math
{Recall}_i = \frac{TP_i}{TP_i + FN_i}
```

### Precision
The $Precision$ of class $i$ is the proportion between correctly classified actual positive instance of the class and all positive instances (correctly and incorrectly classified). The metric is also called *Correctness*, because it is a measure of how well/correct the classificator predicts a certain class.
```math
{Precision}_i = \frac{TP_i}{TP_i + FP_i}
```

### F1-Score
The $F1-Score$ is the harmonic mean between Precision and Recall. 
```math
F1-Score = 2 \cdot \frac{Precision_i \cdot Recall_i}{Precision_i + Recall_i}
```

### Quality
The $Quality$ is another metric that includes both FP and FN. It is also known as the *Jaccard-Index*.
```math
Quality_i = \frac{TP_i}{TP_i + FP_i + FN_i}
```

### Overall Accuracy
The $Overall Accuracy$ is the number of overall true positives compared to all instances.
```math
OA = \frac{\sum_i TP_i}{\sum_i TP_i + FP_i }
```

### Mean Recall
The $Mean Recall$ is another possible metric, where only one metric is used for the whole classification.
```math
Mean Recall = \frac{\sum_i Recall_i}{N}
```

## Visualisation
After classification, the colored point cloud is exported as `.ply` with `helper_functions.py` and visualized in `Meshlab`.

## Extension of the Classification
After Visualization of the classified point cloud in `Meshlab` some problems are noticable, e.g. the ground and facade are classified as each other. This is because of the rotational invariance of the covariance features. Both ground and facade have supposedly high planarity and similar other covariance features. To differentiate them from each other, we can introduce a simple geometric feature. We chose the `height difference` of the neighborhood:

```math
height diff = max(z_{neighbors}) - min(z_{neighbors})
```
By adding this feature to the feature vector the random random forest classification improves noticeably. Specifically the problem with ground and facade classification is drastically improved.

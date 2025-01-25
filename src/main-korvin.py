import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from helper_functions import create_colored_point_cloud, save_colored_point_cloud_as_ply
import h5py
import time



# processing time
start_time = time.time()

#%% --- AUFGABE 1 -------------------------------------------------------------
"""
Laden Sie die gegebene Datei 'point_cloud_data.mat' in Ihrem Python-Skript (z.B. mit dem Modul h5py).
Diese Datei enthält zwei (n x 4)-Matrizen, welche für n 3D-Punkte jeweils die XYZ-Koordinaten sowie
die Klassenzugehörigkeit enthalten. Erstellen Sie eine Funktion, die für eine (n x 3)-Matrix
mit XYZ-Koordinaten für n Punkte als Eingangsgröße einen (n x k)- Vektor der in den jeweiligen
Nachbarschaften enthaltenen Punkte angibt.
"""

# Load the data          
filepath = './data/mat/'
filename = 'point_cloud_data.mat'

with h5py.File(filepath+filename,'r') as file:
    # Extract the training and validation data
    train_data = file['PC_training']    # 4x36932 matrix (x,y,z,class)
    valid_data = file['PC_validation']  # 4x91515 matrix (x,y,z,class)
    
    points3d_train = train_data[:3,:].T # 36932x3 matrix (x,y,z)
    points3d_valid = valid_data[:3,:].T # 91515x3 matrix (x,y,z)

    class_train = train_data[3,:].T     # 36932x1 matrix (class)
    class_valid = valid_data[3,:].T     # 91515x1 matrix (class)
    


valid_data = np.vstack((points3d_valid.T, class_valid)).T
train_data = np.vstack((points3d_train.T, class_train)).T


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
k = 50
indices_neighbors_train, points_neighbors_train = getNeighborhood(points3d_train, k, firstNeighbor=True)
indices_neighbors_valid, points_neighbors_valid = getNeighborhood(points3d_valid, k, firstNeighbor=True)

# Logging
print("==="*30)
print(f"Completed Nearest Neighbors ({round(time.time()-start_time,2)} seconds)\n")
print(f'points_train.shape: {points3d_train.shape}')
print(f'indices_neighbors_train.shape: {indices_neighbors_train.shape}')
print(f'points_neighbors_train.shape: {points_neighbors_train.shape}')

# Time
start_time = time.time()
    
#%% --- AUFGABE 2 -------------------------------------------------------------
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
        # alternativ: eigenvalues[::-1, ::-1] # ist eig. genau was np.flip macht
        
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
cov_features_valid = getCovFeatures(points_neighbors_valid)

# Logging
print("==="*30)
print(f"Completed Covariance Features ({round(time.time()-start_time,2)} seconds)\n")
print(f'cov_features_train.shape: {cov_features_train.shape}')

# Time
start_time = time.time()

#%% --- AUFGABE 3 -------------------------------------------------------------
"""
Führen Sie eine Klassifikation mittels des Random Forest Klassifikators durch. Dieser
Klassifikator soll auf den gekennzeichneten Trainingsdaten trainiert werden, so dass die
Klassifikation der gekennzeichneten Validierungsdaten erfolgen kann.
"""


# mit bootstrap=True und max_samples wird festgelegt, wie viele zufällige Einträge genutzt werden sollen, 
# um die einzelnen Bäume zu trainieren. n_estimators legt fest, wie viele Bäume trainiert werden sollen.
# Die gibt den Random Forest als leeres Konstrukt zurück, welcher dann auf ein Datensatz angewendet wird.
# Werte 20 und 0.4 sind absichtlich niedrig gewählt damit der code schneller läuft. Für "echte" Ergebnisse hochsetzen
rfc = RandomForestClassifier(n_estimators=200,bootstrap=False,n_jobs=4)
rfc = rfc.fit(X=cov_features_train,y=class_train)

# Apply the Random Forest Classifier to the validation data
class_pred = rfc.predict(cov_features_valid)

# Logging
print("==="*30)
print(f"Completed Random Forest Classifier ({round(time.time()-start_time,2)} seconds)\n")

# Time
start_time = time.time()



#%% --- AUFGABE 4 -------------------------------------------------------------
"""
Evaluieren Sie die Güte der erreichten Ergebnisse, indem Sie geeignete Maße über die
Konfusionsmatrix bestimmen. Nutzen Sie auch in den Python-Modulen enthaltene Metriken
und vergleichen Sie diese mit Ihren Ergebnissen aus selbst implementierten Formeln.
"""

# Compute the confusion matrix
cm = confusion_matrix(class_valid, class_pred)

print("Confusion Matrix:")
print(cm)

# save the confusion matrix as a txt file
np.savetxt('confusion_matrix.txt', cm, fmt='%d')

# # scores = cross_val_score(rfc,cov_features_train,class_train,cv=5)
# scores = cross_validate(rfc,cov_features_valid,class_valid) # erlaubt mittels scoring= für eigene Metriken

# # Logging
# print("==="*30)
# print(f"Completed Cross Validation ({round(time.time()-start_time,2)} seconds)\n")
# print(f'scores: {scores}')


# # Step 3 (Optional): Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
# disp.plot(cmap='Blues')  # Adjust the colormap if needed

# # Show the plot
# import matplotlib.pyplot as plt
# plt.show()



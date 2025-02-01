import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from helper_functions import create_colored_point_cloud, save_colored_point_cloud_as_ply
import h5py
import time


# === Here be Functions ===
def getNeighborhood(points: np.ndarray, k: int, excludeFirstNeighbor: bool = False):
    """
    Compute the k nearest neighbors for each point of points.

    Parameters:
        points               : (n x 3)-Matrix with n 3D points (x,y,z)
        k                    : number of neighbors
        excludeFirstNeighbor : if True, exclude the point itself from the neighbors
    Returns:
        indices_neighbors    : (n x k)-Matrix with the indices of the neighbors
        points_neighbors     : (n x k x 3)-Matrix with the coordinates of the neighbors
    """

    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)

    # Find the k+1 nearest neighbors (including the point itself)
    distances, indices = nbrs.kneighbors(points)

    # Exclude the first neighbor (itself) if needed
    if excludeFirstNeighbor:
        indices = indices[:, 1:]

    # Get the coordinates of the neighbors
    points_neighbors = points[indices]

    return indices, points_neighbors

def getCovFeatures(points_neighbors: np.ndarray, flag_dZ: bool=False):
    """
    Compute the covariance features for each point of points_neighbors.
    Covariance features are the 8 entries of the scattermatrix.

    linearity, planarity, scattering, omnivariance, anisotropy,
    eigenentropy, sum of eigenvalues, change of curvature

    Parameters:
        points_neighbors : (n x k x 3)-Matrix with n 3D points and their k neighbors (x,y,z)
        flag_geom        : Boolean for inclusion of height difference in feature vector.
                           False: Only Cov-Features. True: Height-diff is included
    Returns:
        cov_features     : (n x 8)-Matrix with the covariance features
    """

    # Get the number of points
    n = points_neighbors.shape[0]

    # Initialize the matrix to store the covariance features
    if flag_dZ:
        cov_features = np.zeros((n, 9))     # include height difference in feature vector
    else:
        cov_features = np.zeros((n,8))      # dont include height difference in feature vector

    # Compute the covariance features for each point
    for i in range(n):
        # Get the neighbors coordinates (x,y,z) of the point i
        neighbors = points_neighbors[i]

        #! Compute the covariance matrix
        num_neighbors = neighbors.shape[0]           # number of neighbors
        sample_mean   = np.mean(neighbors, axis=0)   # mean of the neighbors
        cov_matrix    = 1/(num_neighbors - 1) * (neighbors - sample_mean).T @ (neighbors - sample_mean) # formula from the script

        #! Compute the eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues    = np.flip(eigenvalues)                    # order largest first
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

        # Store the covariance features (at 0th to 7th column of cov_features)
        cov_features[i,:8] = [linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, sum_eigen, change_curv]

        ###############################################################################################
        if flag_dZ:
            '''
            Als zusätzliches Feature könnte man die Höhendifferenz der Nachbarn berechnen,
            um vor allem Fassade von Boden zu unterscheiden.
            '''
            #! Compute geometric features
            # height difference of the neighbors
            height_diff = np.max(neighbors[:,2]) - np.min(neighbors[:,2])

            # store the geometric features (at 9th column of cov_features)
            cov_features[i,8] = height_diff
        ###############################################################################################

    return cov_features

def evaluateConfusionMatrix(cm, filename_strg=''):
    # Load the confusion matrix
    #cm = np.loadtxt("data/txt/confusion_matrix.txt", dtype=int)
    n_classes = cm.shape[0]

    # Total sum of the matrix
    total = np.sum(cm)

    # Calculate the TP, FP, FN, and TN for each class
    TP = np.zeros(n_classes) # True positive
    FP = np.zeros(n_classes) # False positive
    FN = np.zeros(n_classes) # False negative
    TN = np.zeros(n_classes) # True negative

    for i in range(n_classes):
        TP[i] = cm[i, i]
        FP[i] = np.sum(cm[:, i]) - TP[i]
        FN[i] = np.sum(cm[i, :]) - TP[i]
        TN[i] = total - TP[i] - FP[i] - FN[i]

    # # Print the results
    # for i in range(n_classes):
        #     print(f"Class {i+1}:")
        #     print(f"  TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")


    # Class-wise evaluation metrics
    recall = np.zeros(n_classes)     # completeness
    precision = np.zeros(n_classes)  # correctness
    f1_score = np.zeros(n_classes)
    quality = np.zeros(n_classes)

    for i in range(n_classes):
        recall[i] = TP[i] / (TP[i] + FN[i])
        precision[i] = TP[i] / (TP[i] + FP[i])
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        quality[i] = TP[i] / (TP[i] + FP[i] + FN[i])

    # General evaluation metrics
    overall_accuracy = np.sum(TP) / total  # overall accuracy
    mean_recall = np.mean(recall)          # mean completeness

    # Round all metrics to 2 decimal places
    quality = np.round(quality, 2)
    precision = np.round(precision, 2)
    recall = np.round(recall, 2)
    f1_score = np.round(f1_score, 2)
    overall_accuracy = np.round(overall_accuracy, 2)
    mean_recall = np.round(mean_recall, 2)

    # Print output
    print(f"Quality:          {quality}")
    print(f"Precision:        {precision}")
    print(f"Recall:           {recall}")
    print(f"F1 score:         {f1_score}")
    print(f"Overall accuracy: {overall_accuracy}")
    print(f"Mean recall:      {mean_recall}")

    # Save to file
    output_file = 'data/txt/quality_metrics' + filename_strg + '.txt'

    with open(output_file, 'w') as file:
        print(f"Quality:          {quality}", file=file)
        print(f"Precision:        {precision}", file=file)
        print(f"Recall:           {recall}", file=file)
        print(f"F1 score:         {f1_score}", file=file)
        print(f"Overall accuracy: {overall_accuracy}", file=file)
        print(f"Mean recall:      {mean_recall}", file=file)


# =======================================================================
# === Main Workflow ===
# =======================================================================

# ======================== Data Import ==================================
# processing time
start_overall_timer = time.time()
start_time = time.time()

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

# ==================== Create neighborhoods ============================
k = 50
indices_neighbors_train, points_neighbors_train = getNeighborhood(points3d_train, k, excludeFirstNeighbor=True)
indices_neighbors_valid, points_neighbors_valid = getNeighborhood(points3d_valid, k, excludeFirstNeighbor=True)

# logging
print("==="*30)
print(f"Completed Nearest Neighbors ({round(time.time()-start_time,2)} seconds)\n")
print(f'points_train.shape: {points3d_train.shape}')
print(f'indices_neighbors_train.shape: {indices_neighbors_train.shape}')
print(f'points_neighbors_train.shape: {points_neighbors_train.shape}')

# time
start_time = time.time()

# ================ Create Covariance Features ============================
# Create covariance features in neighborhoodwithout height difference
cov_features_train = getCovFeatures(points_neighbors_train, False)
cov_features_valid = getCovFeatures(points_neighbors_valid, False)

# Create covariance features in neighborhoods WITH height difference
cov_features_train_geom = getCovFeatures(points_neighbors_train, True)
cov_features_valid_geom = getCovFeatures(points_neighbors_valid, True)

# logging
print("==="*30)
print(f"Completed Covariance Features ({round(time.time()-start_time,2)} seconds)\n")
print(f'cov_features_train.shape: {cov_features_train.shape}')
print(f'cov_features_valid.shape: {cov_features_valid.shape}')
print(f'cov_features_train_geom.shape: {cov_features_train_geom.shape}')
print(f'cov_features_valid_geom.shape: {cov_features_valid_geom.shape}')

# time
start_time = time.time()

# =================== Random Forest Classification  ============================
#  Initialize the Random Forest Classifier
## n_estimators: number of trees in the forest
## bootstrap: whether bootstrap samples are used when building trees
#             (False: no bootstrap -> use the whole training set, validation set is used for testing)
## n_jobs: number of jobs to run in parallel (default: 1)
rfc = RandomForestClassifier(n_estimators=200,bootstrap=False,n_jobs=4)
rfc_geom = RandomForestClassifier(n_estimators=200,bootstrap=False,n_jobs=4)

# Train the Random Forest Classifier
rfc = rfc.fit(X=cov_features_train,y=class_train)
rfc_geom = rfc_geom.fit(X=cov_features_train_geom,y=class_train)

# Apply the Random Forest Classifier to the validation data
class_pred = rfc.predict(cov_features_valid)
class_pred_geom = rfc_geom.predict(cov_features_valid_geom)

# logging
print("==="*30)
print(f"Completed Random Forest Classifier ({round(time.time()-start_time,2)} seconds)\n")

# =================== Evaluation with confusion matrix  ============================
# time
start_time = time.time()

# Compute the confusion matrix
cm = confusion_matrix(class_valid, class_pred)
cm_geom = confusion_matrix(class_valid, class_pred_geom)

print("Confusion Matrix:")
print(cm)
print("\nConfusion Matrix (height diff included):")
print(cm_geom)

rowsum = cm.sum(axis=1)
cmp = np.round(cm/rowsum[:,np.newaxis]*100)
print("\nConfusion Matrix in %:")
print(cmp)
# save the confusion matrix as a txt file
np.savetxt('./data/txt/confusion_matrix.txt', cm, fmt='%d')
np.savetxt('./data/txt/confusion_matrix_percent.txt', cmp, fmt='%d')

rowsum_geom = cm_geom.sum(axis=1)
cmp_geom = np.round(cm_geom/rowsum_geom[:,np.newaxis]*100)
print("\nConfusion Matrix (height diff included) in %:")
print(cmp_geom)
# save the confusion matrix as a txt file
np.savetxt('./data/txt/confusion_matrix_geom.txt', cm_geom, fmt='%d')
np.savetxt('./data/txt/confusion_matrix_percent_geom.txt', cmp_geom, fmt='%d')

# =================== Plot the confusion matrices  ============================
display_labels=["wire", "pole/trunk", "facade", "ground", "vegetation"]

# Without normalization -> shows NUMBER of points in each confusion matrix row/col
dispCM = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred, display_labels = display_labels)
plt.title("Konfusionsmatrix")
plt.savefig('./data/img/confusion_matrix.png')
dispCM_geom = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred_geom, display_labels = display_labels)
plt.title("Konfusionsmatrix (mit Höhendifferenz)")
plt.savefig('./data/img/confusion_matrix_geom.png')

# With normalization to all -> shows PERCENTAGE of points in each confusion matrix row/col
dispCM = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred, display_labels = display_labels, normalize = 'all')
plt.title("Konfusionsmatrix normalisiert nach All")
plt.savefig('./data/img/confusion_matrix_normAll.png')
dispCM_geom = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred_geom, display_labels = display_labels, normalize = 'all')
plt.title("Konfusionsmatrix (mit Höhendifferenz) normalisiert nach All")
plt.savefig('./data/img/confusion_matrix_geom_normAll.png')

# With normalization to true -> shows in percent to which class points of a certain class are predicted
# Examples for no geom:
#       - 95% of ground points are actually classified as ground
#       - 46% of facade points are actually classified as as facade, but 32% of them are classified as ground
dispCM = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred, display_labels = display_labels, normalize = 'true')
plt.title("Konfusionsmatrix normalisiert nach True")
plt.savefig('./data/img/confusion_matrix_normTrue.png')
dispCM_geom = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred_geom, display_labels = display_labels, normalize = 'true')
plt.title("Konfusionsmatrix (mit Höhendifferenz) normalisiert nach True")
plt.savefig('./data/img/confusion_matrix_geom_normTrue.png')

# With normalization to pred -> shows in percent what classes contributed to a certain predicted class
# Examples for no geom:
#       - 61% of the points classified as wire were actually facade points.
#       - 31% of the points classified as facade were actually ground points.
dispCM = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred, display_labels = display_labels, normalize = 'pred')
plt.title("Konfusionsmatrix normalisiert nach Pred")
plt.savefig('./data/img/confusion_matrix_normPred.png')
dispCM_geom = ConfusionMatrixDisplay.from_predictions(class_valid, class_pred_geom, display_labels = display_labels, normalize = 'pred')
plt.title("Konfusionsmatrix (mit Höhendifferenz) normalisiert nach Pred")
plt.savefig('./data/img/confusion_matrix_geom_normPred.png')

# logging
print("==="*30)
print(f"Completed Evaluation with Confusion Matrix ({round(time.time()-start_time,2)} seconds)\n")

# =================== Evaluation with further metrics  ============================
# time
start_time = time.time()

# Print and save other Evaluation Metrics
print("\nFurther Quality metrics:")
evaluateConfusionMatrix(cm,'')
print("\nFurther Quality metrics (height diff included):")
evaluateConfusionMatrix(cm_geom,'_geom')

# =================== Evaluation with classification_report ===========================
print("\n")
print("Classification report:")
classificationReport = classification_report(class_valid, class_pred, target_names = display_labels)
print(classificationReport)
with open('./data/txt/quality_metrics.txt', 'a') as f:
    f.write("\n\nClassification report:")
    f.write(classificationReport)

print("Classification report (include height difference):")
classificationReportGeom = classification_report(class_valid, class_pred_geom,  target_names = display_labels)
print(classificationReportGeom)
with open('./data/txt/quality_metrics_geom.txt', 'a') as f:
    f.write("\n\nClassification report (no height difference):")
    f.write(classificationReportGeom)

# logging
print("==="*30)
print(f"Completed Evaluation with Classification Report ({round(time.time()-start_time,2)} seconds)\n")

# =================== Create and save colored point clouds  ============================
# time
start_time = time.time()

# Prediction
colored_point_cloud = create_colored_point_cloud(valid_data, class_pred)
save_colored_point_cloud_as_ply(colored_point_cloud, 'valid_pred')
# Prediction with geom feature
colored_point_cloud = create_colored_point_cloud(valid_data, class_pred_geom)
save_colored_point_cloud_as_ply(colored_point_cloud, 'valid_pred_geom')
# Ground truth
colored_point_cloud = create_colored_point_cloud(valid_data, class_valid)
save_colored_point_cloud_as_ply(colored_point_cloud, 'valid')
# Train data
colored_point_cloud = create_colored_point_cloud(train_data, class_train)
save_colored_point_cloud_as_ply(colored_point_cloud, 'train')

# logging
print("==="*30)
print(f"Completed Saving Point Clouds ({round(time.time()-start_time,2)} seconds)\n")
print("To visualize point clouds drag and drop .ply Files in MeshLab")
print("==="*30)
print(f"Completed Total Workflow ({round(time.time()-start_overall_timer,2)} seconds)")
print("==="*30)
print("==="*30)

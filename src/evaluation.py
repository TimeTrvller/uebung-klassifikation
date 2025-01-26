import numpy as np

# Load the confusion matrix
cm = np.loadtxt("confusion_matrix.txt", dtype=int)
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
overall_accuracy = np.sum(TP) / total

print(overall_accuracy)
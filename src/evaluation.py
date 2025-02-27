import numpy as np
import json
import matplotlib.pyplot as plt

# # Load the confusion matrix
# cm = np.loadtxt("data/txt/confusion_matrix.txt", dtype=int)
# n_classes = cm.shape[0]

# # Total sum of the matrix
# total = np.sum(cm)

# # Calculate the TP, FP, FN, and TN for each class
# TP = np.zeros(n_classes) # True positive
# FP = np.zeros(n_classes) # False positive
# FN = np.zeros(n_classes) # False negative
# TN = np.zeros(n_classes) # True negative

# for i in range(n_classes):
#     TP[i] = cm[i, i]
#     FP[i] = np.sum(cm[:, i]) - TP[i]
#     FN[i] = np.sum(cm[i, :]) - TP[i]
#     TN[i] = total - TP[i] - FP[i] - FN[i]

# # # Print the results
# # for i in range(n_classes):
# #     print(f"Class {i+1}:")
# #     print(f"  TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")


# # Class-wise evaluation metrics
# recall = np.zeros(n_classes)     # completeness
# precision = np.zeros(n_classes)  # correctness
# f1_score = np.zeros(n_classes)
# quality = np.zeros(n_classes)

# for i in range(n_classes):
#     recall[i] = TP[i] / (TP[i] + FN[i])
#     precision[i] = TP[i] / (TP[i] + FP[i])
#     f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
#     quality[i] = TP[i] / (TP[i] + FP[i] + FN[i])
    
# # General evaluation metrics
# overall_accuracy = np.sum(TP) / total  # overall accuracy
# mean_recall = np.mean(recall)          # mean completeness

# print(f"Quality:          {quality}")
# print(f"Precision:        {precision}")
# print(f"Recall:           {recall}")
# print(f"F1 score:         {f1_score}")
# print(f"Overall accuracy: {overall_accuracy}")
# print(f"Mean recall:      {mean_recall}")


# load classification_reports.json
with open("data/json/classification_reports_max_depth.json", "r") as f:
    classification_reports = json.load(f)
    
# plot the "weighted avg" "f1-score"

f1_scores = []
for report in classification_reports:
    f1_scores.append(report["weighted avg"]["f1-score"])
baumtiefe = [i for i in range(1, 21)]
    
plt.figure()
plt.plot(baumtiefe, f1_scores)
plt.xlabel("Baumtiefe")
plt.ylabel("F1 score")
plt.title("F1 score für verschiedene Baumtiefen")
plt.show()
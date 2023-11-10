import numpy as np
def f1_score(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = sum(confusion_matrix[j, i] for j in range(num_classes) if j != i)
        fn = sum(confusion_matrix[i, j] for j in range(num_classes) if j != i)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_scores[i] = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    macro_f1 = np.mean(f1_scores)
    return f1_scores, macro_f1

# Example 3x3 confusion matrix
confusion = np.array([[ 80,  90,   2],[  1 , 146 ,55 ],[ 11, 22 ,634  ]])
f1_scores, macro_f1 = f1_score(confusion)
print("F1-scores for each class:", f1_scores)
print("Macro F1-score:", macro_f1)
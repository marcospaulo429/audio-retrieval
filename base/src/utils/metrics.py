def precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def recall(true_positives, false_negatives):
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def accuracy(true_positives, true_negatives, false_positives, false_negatives):
    total = true_positives + true_negatives + false_positives + false_negatives
    if total == 0:
        return 0.0
    return (true_positives + true_negatives) / total
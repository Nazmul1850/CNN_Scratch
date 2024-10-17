import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate accuracy
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))*100.

def precision(y_true, y_pred):
    """
    Calculate precision
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-8)
    return precision


def recall(y_true, y_pred):
    """
    Calculate recall
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-8)
    return recall


def f1_score(y_true, y_pred):
    """
    Calculate f1 score
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+1e-8))


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    """
    return np.array([[np.sum(np.logical_and(y_pred == 0, y_true == 0)), np.sum(np.logical_and(y_pred == 0, y_true == 1))],
                     [np.sum(np.logical_and(y_pred == 1, y_true == 0)), np.sum(np.logical_and(y_pred == 1, y_true == 1))]])


def macrof1_score(y_true, y_pred):
    """
    Calculate macro f1 score
    """
    return np.mean([f1_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
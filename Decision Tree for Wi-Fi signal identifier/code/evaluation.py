import numpy as np


def predict_label(test_dataset, tree):
    """
    Predict room labels for test dataset using trained decision tree.
    
    Args:
        test_dataset : numpy array of shape (n_samples, n_attributes + 1)
                       last column contains true labels - not used here
        tree         : dict, trained decision tree from decision_tree_learning()
        
    Returns:
        predicted_labels: numpy array of predicted labels for each sample
    """

    if len(test_dataset) == 0:
        return np.array([])
    if tree is None:
        raise ValueError("Tree cannot be None")

    predicted_labels = np.zeros_like(test_dataset[:, 0])
    
    for idx, sample in enumerate(test_dataset):
        current_node = tree

        while "leaf" not in current_node:
            attribute = current_node["attribute"]
            value = current_node["value"]

            if sample[attribute] < value:
                current_node = current_node["left"]
            else:
                current_node = current_node["right"]

        predicted_labels[idx] = current_node["leaf"]

    return predicted_labels


def confusion_matrix(test_dataset, predicted_labels):
    """
    Build a confusion matrix based on predicted and actual results.

    Args:
        test_dataset     : numpy array of shape (n_samples, n_attributes + 1)
                           last column contains true labels - all that we use here
        predicted_labels : numpy array of predicted labels for each sample

    Returns:
        conf_matrix: square numpy matrix showing distribution between true and predicted labels
    """
    true_labels = test_dataset[:, -1]
    all_labels = np.unique(np.concatenate((true_labels, predicted_labels)))
    n_labels = len(all_labels)

    conf_matrix = np.zeros((n_labels, n_labels), dtype = float)
    for i in range(n_labels):
        for j in range(n_labels):
            conf_matrix[i, j] = np.sum((test_dataset[:, -1] == all_labels[i]) & \
                                      (predicted_labels == all_labels[j]))

    return conf_matrix


def get_accuracy(conf_matrix):
    """ Calculate the accuracy using confusion matrix (float) """
    return np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)


def get_precision(conf_matrix):
    """ Calculate the precision of each class """
    precisions = np.zeros_like(conf_matrix[0], dtype=float)
    for i in range(len(precisions)):
        if np.sum(conf_matrix[:, i]) == 0:
            precisions[i] = 0
        else:
            precisions[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i])
    return precisions


def get_recall(conf_matrix): 
    """ Calculate the recall of each class """
    recalls = np.zeros_like(conf_matrix[0], dtype=float)
    for i in range(len(recalls)):
        if np.sum(conf_matrix[i]) == 0:
            recalls[i] = 0
        else:
            recalls[i] = conf_matrix[i, i] / np.sum(conf_matrix[i])
    return recalls


def get_f1_score(precision, recall):
    """ Calculate the F1 score of each class """
    if sum(precision + recall) == 0:
        raise ValueError("Can't calculate f1 score with 0 precision and recall")
    return 2 * precision * recall / (precision + recall)

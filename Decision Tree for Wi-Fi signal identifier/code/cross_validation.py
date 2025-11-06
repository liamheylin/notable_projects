import numpy as np
from numpy.random import default_rng
from tree_builder import decision_tree_learning
from evaluation import predict_label, confusion_matrix, get_accuracy, get_precision, get_recall, get_f1_score
from pruning import prune_tree, calculate_tree_depth


def k_fold_split(n_splits, n_instances, random_generator):
    """
    Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits         : int, number of splits
        n_instances      : int, number of instances to split
        random_generator : np.random.generator, a random generator

    Returns:
        list: a list (length n_splits). Each element in the list contains a
            numpy array giving the indices of the instances in that split.
    """
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices


def train_test_valid_k_fold(k_folds, dataset, seed=42):
    """
    Generate train and test indices for each fold.
    
    Args:
        k_folds : int, number of folds
        dataset : numpy array, dataset to be split
        seed    : int, number to be used for the seed for the random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a dictionary
            with two keys:
                - "test_indices"  : numpy array containing the test indices (length is e.g. 200)
                - "train_indices" : numpy array containing the combined train and validation indices (length is e.g. 1800)
    """
    random_generator = default_rng(seed)

    split_indices = k_fold_split(k_folds, len(dataset), random_generator)
    all_folds = []
    
    for k in range(k_folds):
        test_indices = split_indices[k]
        train_indices = np.concatenate(split_indices[:k] + split_indices[(k+1):])
        all_folds.append({"test_indices":test_indices, "train_indices":train_indices})

    return all_folds


def get_pre_pruned_results(dataset, k_folds):
    """
    Perform k-fold cross-validation, evaluating the decision tree for each fold.

    Args:
        dataset : numpy array containing the dataset to be evaluated
        k_folds : int, number of folds to use for cross-validation

    Returns:
        dict: dictionary containing the following keys:
            - "conf_matrix" : numpy array representing the confusion matrix for all folds
            - "accuracy"    : float representing the overall accuracy across all folds
            - "precision"   : numpy array containing precision values for each class
            - "recall"      : numpy array containing recall values for each class
            - "f1_score"    : numpy array containing F1 scores for each class
            - "avg_depth"   : float representing the average depth of the trees for all folds
    """
    conf_matrices = []
    depth_counts = []
    all_folds_indices = train_test_valid_k_fold(k_folds, dataset)
    
    for i in range(k_folds):
        training_indices, test_indices = all_folds_indices[i]["train_indices"], all_folds_indices[i]["test_indices"]
        training_dataset, test_dataset = dataset[training_indices], dataset[test_indices]

        node, max_depth = decision_tree_learning(training_dataset, 0)

        predicted_labels = predict_label(test_dataset, node)

        conf_matrix = confusion_matrix(test_dataset, predicted_labels)

        conf_matrices.append(conf_matrix)
        depth_counts.append(max_depth)

    final_conf_matrix = np.sum(conf_matrices, axis = 0) / k_folds
    avg_depth = np.sum(depth_counts) / k_folds

    accuracy = get_accuracy(final_conf_matrix)
    precision = get_precision(final_conf_matrix)
    recall = get_recall(final_conf_matrix)
    f1_score = get_f1_score(precision, recall)

    dictionary = {"conf_matrix": final_conf_matrix, 
                  "accuracy": accuracy,
                  "precision": precision,
                  "recall": recall,
                  "f1_score": f1_score,
                  "avg_depth": avg_depth}

    return dictionary


def get_pruned_results(dataset, k_folds):
    """
    Perform k-fold cross-validation, evaluating the decision tree for each fold.

    Args:
        dataset : numpy array containing the dataset to be evaluated
        k_folds : int, number of folds to use for cross-validation

    Returns:
        dict: dictionary containing the following keys:
            - "conf_matrix" : numpy array representing the confusion matrix for all folds
            - "accuracy"    : float representing the overall accuracy across all folds
            - "precision"   : numpy array containing precision values for each class
            - "recall"      : numpy array containing recall values for each class
            - "f1_score"    : numpy array containing F1 scores for each class
            - "avg_depth"   : float representing the average depth of the trees for all folds
    """
    n_labels = len(np.unique(dataset[:, -1]))
    total_conf_matrix = np.zeros((n_labels, n_labels))
    total_depth = 0

    all_folds_indices = train_test_valid_k_fold(k_folds, dataset)
    
    for i in range(k_folds):
        training_indices, test_indices = all_folds_indices[i]["train_indices"], all_folds_indices[i]["test_indices"]
        training_dataset, test_dataset = dataset[training_indices], dataset[test_indices]

        inner_folds_indices = train_test_valid_k_fold(k_folds-1, training_dataset)

        for j in range(k_folds-1):
            inner_train_indices, validation_indices = inner_folds_indices[j]["train_indices"], inner_folds_indices[j]["test_indices"]
            inner_train_dataset, validation_dataset = training_dataset[inner_train_indices], training_dataset[validation_indices]

            node, max_depth = decision_tree_learning(inner_train_dataset, 0)

            pruned_tree = prune_tree(node, inner_train_dataset, validation_dataset)
            depth = calculate_tree_depth(pruned_tree)
            labels = predict_label(test_dataset, pruned_tree)
            conf_matrix = confusion_matrix(test_dataset, labels)

            total_conf_matrix += conf_matrix
            total_depth += depth

    final_conf_matrix = total_conf_matrix / (k_folds * (k_folds-1))
    avg_depth = total_depth / (k_folds * (k_folds-1))

    accuracy = get_accuracy(final_conf_matrix)
    precision = get_precision(final_conf_matrix)
    recall = get_recall(final_conf_matrix)
    f1_score = get_f1_score(precision, recall)

    dictionary = {"conf_matrix": final_conf_matrix, 
                  "accuracy": accuracy,
                  "precision": precision,
                  "recall": recall,
                  "f1_score": f1_score,
                  "avg_depth": avg_depth}

    return dictionary

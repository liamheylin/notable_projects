import numpy as np


def calculate_entropy(dataset):
    """ Calculate entropy (float) """
    if len(dataset) == 0:
        return 0.0
    labels = dataset[:, -1]

    #generate counts and then probabilities for each label
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    entropy = -np.sum(probabilities * np.log2(probabilities))
        
    return entropy


def calculate_remainder(dataset_left, dataset_right):
    """
    Calculate weighted entropy after split.
    
    Args:
        dataset_left  : numpy array for left partition
        dataset_right : numpy array for right partition
        
    Returns:
        float: weighted average of entropies
    """
    n_left = len(dataset_left)
    n_right = len(dataset_right)

    if n_left == 0 and n_right == 0:
        return 0.0

    #calculate weights
    weight_left = n_left / (n_left + n_right)
    weight_right = n_right / (n_left + n_right)

    #calculate entropies
    entropy_left = calculate_entropy(dataset_left)
    entropy_right = calculate_entropy(dataset_right)

    remainder = weight_left * entropy_left + weight_right * entropy_right

    return remainder
                

def split_on_attribute(dataset, attribute_idx, value):
    """
    Split dataset based on optimal attributes and threshold values.
    
    Args:
        dataset       : numpy array
        attribute_idx : int, index of attribute to split on (WiFi point)
        value         : float, threshold value for split (WiFi strength)
        
    Returns:
        tuple: (left_dataset, right_dataset)
               left contains samples where WiFi strength < value
               right contains samples where WiFi strength >= value
    """
    dataset_left = dataset[dataset[:, attribute_idx] < value]
    dataset_right = dataset[dataset[:, attribute_idx] >= value]    
        
    return dataset_left, dataset_right


def find_split(training_dataset):
    """
    Find the best attribute and value to split on using information gain.
    
    Args:
        training_dataset : numpy array of shape (n_samples, n_attributes + 1)
                           last column contains labels
        
    Returns:
        tuple: (best_attribute_idx, best_value)
    """
    if len(training_dataset) == 0:
        raise ValueError("Cannot find split on empty dataset")
    
    lowest_remainder = np.inf
    best_attribute_idx, best_value = None, None
    n_attributes = np.shape(training_dataset)[1] - 1
    
    #search over each attribute and within that each potential split point between values to find the lowest remainder
    for attribute_idx in range(n_attributes):
        values = np.unique(training_dataset[:, attribute_idx])
        for value_idx in range(len(values) - 1):
            value = (values[value_idx] + values[value_idx + 1]) / 2
            dataset_left, dataset_right = split_on_attribute(training_dataset, attribute_idx,value)
            if len(dataset_left) == 0 or len(dataset_right) == 0:
                continue
            remainder = calculate_remainder(dataset_left, dataset_right)
            if remainder < lowest_remainder:
                lowest_remainder = remainder
                best_attribute_idx, best_value = attribute_idx, value
    
    if best_attribute_idx is None or best_value == None:
        raise ValueError("No valid split found")

    return best_attribute_idx, best_value


def decision_tree_learning(dataset, depth=0):
    """
    Build the decision tree.
    
    Args:
        dataset : numpy array of shape (n_samples, n_attributes + 1)
                  last column contains labels
        depth   : int, current depth in tree
        
    Returns:
        tuple: (node, max_depth_reached)
               node is a dict representing the tree structure
               max_depth_reached is the deepest level in this subtree
    """
    if len(dataset) == 0:
        raise ValueError("Cannot build tree from empty dataset")

    labels = dataset[:, -1]
    if len(np.unique(labels)) == 1:
        return ({"leaf": labels[0]}, depth)
        
    attribute, value = find_split(dataset)
    dataset_left, dataset_right = split_on_attribute(dataset, attribute, value)
    
    left_branch, left_depth = decision_tree_learning(dataset_left, depth + 1)
    right_branch, right_depth = decision_tree_learning(dataset_right, depth + 1)
        
    node = {"attribute" : attribute,
            "value"     : value,
            "left"      : left_branch,
            "right"     : right_branch
                }
    
    return (node, max(left_depth, right_depth))

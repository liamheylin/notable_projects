import numpy as np
from evaluation import predict_label


def calculate_validation_error(node, validation_dataset):
    """
    Calculate the number of incorrect predictions on validation dataset.
    
    Args:
        node               : the decision tree node to evaluate
        validation_dataset : numpy array, dataset used for validation
        
    Returns:
        int: number of incorrect predictions
    """
    predictions = predict_label(validation_dataset, node)
    true_labels = validation_dataset[:, -1]
    errors = np.sum(predictions != true_labels)
    return errors


def label_new_node(node, training_dataset):
    """
    Assign majority label to a node based on training data.
    
    Args:
        node             : the decision tree node to label
        training_dataset : numpy array, dataset used for determining majority label
        
    Returns:
        int: the majority label for the node
    """
    predictions = predict_label(training_dataset, node)
    unique_labels, counts = np.unique(predictions, return_counts=True)
    new_label = unique_labels[np.argmax(counts)]
    return new_label


def should_prune(node, training_dataset, validation_dataset):
    """
    Determine if a node should be pruned based on validation error.
    
    Compares validation error before and after pruning. Pruning is beneficial
    if the error after pruning is less than or equal to the error before pruning.
    
    Args:
        node               : the decision tree node to evaluate for pruning
        training_dataset   : numpy array, dataset used for determining majority label
        validation_dataset : numpy array, dataset used for error calculation
        
    Returns:
        bool: True if pruning reduces or maintains validation error
    """
    # Find current error
    current_error = calculate_validation_error(node, validation_dataset)
    
    # Find pruned error
    majority_label = label_new_node(node, training_dataset)
    pruned_version = {"leaf": majority_label}
    new_error = calculate_validation_error(pruned_version, validation_dataset)
    
    return new_error <= current_error

def cut_branch(node, training_dataset):
    """
    Convert a node with branches into a leaf node.
    
    Args:
        node             : the decision tree node to convert
        training_dataset : numpy array, dataset used for determining majority label
        
    Returns:
        dict: a leaf node with the majority label
    """
    majority_label = label_new_node(node, training_dataset)
    return {"leaf": majority_label}


def start_pruning(tree, training_dataset, validation_dataset):
    """
    Recursively prune a decision tree starting from the bottom.
    
    Apply bottom-up pruning by first attempting to prune subtrees, 
    then checking if the current node can be pruned after its
    branches have been processed.
    
    At each recursion step, only the subset of training and validation
    data that reaches the current node is passed down to its children.

    
    Args:
        tree               : the decision tree to prune
        training_dataset   : numpy array, dataset used for determining majority labels
        validation_dataset : numpy array, dataset used for error calculation
        
    Returns:
        tuple: (pruned_tree, was_pruned) where was_pruned indicates if any pruning occurred
    """
    if "leaf" in tree:
        return tree, False

    # Split current datasets by this node's split
    attr = tree["attribute"]
    val = tree["value"]
    left_train = training_dataset[training_dataset[:, attr] < val]
    right_train = training_dataset[training_dataset[:, attr] >= val]
    left_val = validation_dataset[validation_dataset[:, attr] < val]
    right_val = validation_dataset[validation_dataset[:, attr] >= val]

    # Recurse on children with their subsets
    tree["left"], left_pruned = start_pruning(tree["left"], left_train, left_val)
    tree["right"], right_pruned = start_pruning(tree["right"], right_train, right_val)

    # After children processed, try pruning this node using the node's own subsets
    if "leaf" in tree["left"] and "leaf" in tree["right"]:
        if should_prune(tree, training_dataset, validation_dataset):
            pruned_node = {"leaf": label_new_node(tree, training_dataset)}
            return pruned_node, True

    return tree, (left_pruned or right_pruned)
    

def prune_tree(tree, training_set, validation_set):
    """
    Prune a decision tree iteratively until no beneficial pruning remains.
    
    Repeatedly applies pruning to the tree until no more nodes can be
    beneficially pruned (i.e., pruning no longer reduces validation error).
    
    Args:
        tree           : the decision tree to prune
        training_set   : numpy array, dataset used for determining majority labels
        validation_set : numpy array, dataset used for error calculation
        
    Returns:
        dict: the fully pruned decision tree
    """
    pruned = True
    while pruned:
        tree, pruned = start_pruning(tree, training_set, validation_set)
    return tree

def calculate_tree_depth(tree):
    """
    Calculate the maximum depth of a tree structure after pruning.
    
    Args:
        tree : dict representing the tree structure
        
    Returns:
        int: maximum depth of the tree
    """
    if "leaf" in tree:
        return 0
    
    left_depth = calculate_tree_depth(tree["left"])
    right_depth = calculate_tree_depth(tree["right"])
    
    return 1 + max(left_depth, right_depth)
import numpy as np
import matplotlib.pyplot as plt
from cross_validation import get_pre_pruned_results, get_pruned_results


def save_confusion_matrix_image(confusion_matrix, filename, title):
    """Save a png file of the confusion matrix with the given filename and title.
    Args:
    confusion_matrix: the confusion matrix for which you wish to create a visualisation
    filename: the name you wish to save the image as
    title: the title to be placed in the image.
    Returns:
    None
    """

    # Axis labels
    labels = ["Room 1", "Room 2", "Room 3", "Room 4"]

    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix, cmap="RdYlGn", vmin=0, vmax=50)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black")

    # Set tick positions and labels and move x-axis ticks and labels to the top
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    
    # Set lables and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename)


def save_results_table(measures_table, filename, title):
    """Save a png file of the results table with the given filename and title.
    Args:
    measures_table: the results table for which you wish to create a visualisation
    filename: the name you wish to save the image as
    title: the title to be placed in the image.
    Returns:
    None
    """
    # Axis labels
    x_labels = ["Recall", "Precision", "F1 Measure"]
    y_labels = ["Room 1", "Room 2", "Room 3", "Room 4"]

    fig, ax = plt.subplots()
    ax.imshow(measures_table, cmap="RdYlGn", vmin=0.7, vmax=1)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(measures_table):
        ax.text(j, i, f"{val:.1%}", ha="center", va="center", color="black")

    # Set tick positions and labels and move x-axis ticks and labels to the top
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    
    # Set lables and title
    ax.set_ylabel("Measure per class")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename)


def process_dataset_results(dataset, dataset_name, is_pruned, k_folds=10):
    """Process results for a single dataset and save visualizations.
    
    Args:
        dataset: The dataset to process
        dataset_name: Name of the dataset ('clean' or 'noisy')
        is_pruned: Boolean indicating whether to use pruned results
        k_folds: Number of folds for cross-validation
    """
    # Get results based on pruning status
    if is_pruned:
        results = get_pruned_results(dataset, k_folds)
        pruning_status = "After Pruning"
    else:
        results = get_pre_pruned_results(dataset, k_folds)
        pruning_status = "Before Pruning"
    
    # Generate filenames and titles
    pruning_prefix = "pruned" if is_pruned else "pre_pruned"
    conf_matrix_filename = f"visualisations/confusion_matrices/{pruning_prefix}_{dataset_name}_confusion_matrix.png"
    results_table_filename = f"visualisations/results_tables/{pruning_prefix}_{dataset_name}_results_table.png"
    
    conf_matrix_title = f"Confusion Matrix: {dataset_name.title()} Data {pruning_status}"
    results_table_title = f"Measures per class: {dataset_name.title()} Data {pruning_status}"
    
    # Save confusion matrix
    save_confusion_matrix_image(results["conf_matrix"], conf_matrix_filename, conf_matrix_title)
    
    # Print accuracy and average depth for each case
    print("--------------------------------")
    print(f"The {pruning_prefix} {dataset_name} results are:")
    print("--------------------------------")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"Average depth: {results['avg_depth']:.1f}")
    print()
    
    # Create and save measures table
    measures_table = np.column_stack((results["recall"], results["precision"], results["f1_score"]))
    save_results_table(measures_table, results_table_filename, results_table_title)


def run_results(clean_dataset, noisy_dataset):
    """Kick off the functions to generate trees 
        and save images containing the results."""
    process_dataset_results(clean_dataset, "clean", is_pruned=False)
    process_dataset_results(noisy_dataset, "noisy", is_pruned=False)
    process_dataset_results(clean_dataset, "clean", is_pruned=True)
    process_dataset_results(noisy_dataset, "noisy", is_pruned=True)
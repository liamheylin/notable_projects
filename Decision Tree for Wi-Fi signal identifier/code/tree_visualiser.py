import random
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_tree(tree, title="Decision Tree Visualisation", 
              depth=0, x=0, y=0, dx=0.2, dy=0.4, ax=None):
    """
    Plot the decision tree.

    Args:
        tree   : dict, decision tree produced by decision_tree_learning()
        title  : string, title of the plot
        depth  : int, current depth level
        x, y   : float, current coordinates for node placement
        dx, dy : float, horizontal and vertical spacing parameters
        ax     : matplotlib.axes.Axes, axes to plot on

    Returns:
        ax     : matplotlib.axes.Axes, object containing the drawn decision tree
    """
    # Create a single figure for the decision tree
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.set_title(title)

    # Draw leaf nodes
    if "leaf" in tree:
        ax.text(x, y, f"leaf:{tree['leaf']:.3f}", ha="center", va="center", fontsize=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="lightblue"))
        return

    # Draw internal nodes
    node_label = f"[X{tree['attribute']} < {tree['value']:.1f}]"
    ax.text(x, y, node_label, ha="center", va="center", fontsize=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="lightblue"))

    # Compute children placement
    left_x = x - dx / (((depth+1) * (depth+1)) +1.5)
    right_x = x + dx / (((depth+1) * (depth+1)) +1.5)
    next_y = y - dy

    # Draw branches
    named_colors = list(colors.TABLEAU_COLORS.keys())
    color = random.choice(named_colors)
    ax.plot([x, left_x], [y - 0.01, next_y + 0.05], color=color, linewidth=1.0)
    ax.plot([x, right_x], [y - 0.01, next_y + 0.05], color=color, linewidth=1.0)

    # Build left and right subtree plots
    plot_tree(tree["left"], title, depth + 1, left_x, next_y, dx, dy, ax)
    plot_tree(tree["right"], title, depth + 1, right_x, next_y, dx, dy, ax)
    
    plt.tight_layout()

    return ax

def save_tree_image(tree, title, filename):
    """ Save image of a decision tree visualisation """
    ax = plot_tree(tree, title)
    plt.savefig(filename, dpi=300)


def visualise_clean_tree():
    """ Visualise a decision tree trained on the entire clean dataset """
    from data_loader import load_dataset
    from tree_builder import decision_tree_learning
    
    clean_dataset = load_dataset("wifi_db/clean_dataset.txt")
    tree, depth = decision_tree_learning(clean_dataset)
    title = "Visualisation of a Decision Tree Trained on the Entire Clean Dataset"
    save_tree_image(tree, title, "visualisations/clean_tree_visualisation.png")

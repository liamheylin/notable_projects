import os
from data_loader import load_dataset
from results import run_results
from tree_visualiser import visualise_clean_tree


def create_directories():
    """ Create directories within the current directory to store the images for the report """
    os.makedirs("visualisations/confusion_matrices", exist_ok=True)
    os.makedirs("visualisations/results_tables", exist_ok=True)

if __name__ == "__main__":
    create_directories()
    clean_dataset = load_dataset("wifi_db/clean_dataset.txt")
    noisy_dataset = load_dataset("wifi_db/noisy_dataset.txt")
    run_results(clean_dataset, noisy_dataset)
    visualise_clean_tree()

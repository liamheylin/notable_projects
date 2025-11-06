import numpy as np


def load_dataset(filepath):
    """ Load dataset from text file """
    dataset = []
    try:
        with open(filepath, "r") as datasetfile:
            for line in datasetfile:
                line = line.strip()
                if line:
                    row = list(map(float, line.split()))
                    dataset.append(row)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    except ValueError as e:
        raise ValueError(f"Invalid data format in file: {e}")
    
    return np.array(dataset)

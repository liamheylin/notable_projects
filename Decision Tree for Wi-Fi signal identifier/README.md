# ML CW1



## How to run the code on the lab machines

1. To obtain the code, download and unpack the zip file, or clone the GitLab repository using SSH into the home directory:

    ```git clone git@gitlab.doc.ic.ac.uk:ljh25/ml-cw1.git```

2. Activate virtual environment using:

    ```source /vol/lab/ml/intro2ml/bin/activate```

3. Change the working directory to the directory containing the project repository:

    ```cd ml-cw1```

4. Run the code using:
    
    ```python code/main.py```

## Outputs

* The code prints accuracy and depth metrics for each of the 4 combinations pruned/not pruned, clean/noisy
* The code generates visualisations used in the report stored in the visualisations directory as .png files

## Code structure

```
- main.py                        highest level script to run all code
    - tree_visualiser.py         creates visualisation on full clean dataset
    - load_dataset.py            loads the dataset
    - results.py                 creates all the images for the report
        - cross_validation.py    splits the datasets and creates, evaluates and prunes multiple trees
            - tree_builder.py    builds a decision tree
            - evaluation.py      generates evaluation metrics for an individual tree
            - pruning.py         prunes a tree
```

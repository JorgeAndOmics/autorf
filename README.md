
# AutoRF

A Python script to implement a Random Forest approach for bioinformatics using a single function with parameters as arguments. The script takes in a dataset and a set of parameters, and uses the Random Forest algorithm to train and test a model on the dataset. It also includes functions to plot the feature importances, decision surface and export the results in a CSV file.

## Requirements

-   Pandas
-   Numpy
-   Scikit-learn
-   Matplotlib
-   Seaborn

## Usage

The script can be run from the command line using the following command:

    python autorf.py --dataset dataset_file.csv --target target_variable_name 

where `dataset_file.csv` is the path to the dataset file and `target_variable_name` is the name of the target variable.

## Optional Arguments

The following optional arguments can be passed to the script:

-   `--n_trees`: The number of trees in the forest (default value = 100).
-   `--max_depth`: The maximum depth of each tree (default value = None).
-   `--min_samples_split`: The minimum number of samples required to split an internal node (default value = 2).
-   `--min_samples_leaf`: The minimum number of samples required to be a leaf node (default value = 1).
-   `--max_features`: The number of features to consider when looking for the best split (default value = "auto").
-   `--criterion`: The criterion for measuring the quality of a split (default value = "gini").
-   `--sample_selection`: The method for selecting samples for training each tree (default value = "auto").

## Output

The script generates the following outputs:

-   A feature importance plot that shows the relative importance of each feature in the model.
-   A decision surface plot (if the dataset has only two features) that shows the decision boundary of the model.
-   A CSV file named `results.csv` that contains the results of the analysis.

## Batch Processing

The script also includes a function `batch_process` that can perform a batch process, in which multiple datasets can be processed at the same time. The function takes in a list of datasets and a list of target variables, and returns a list of trained Random Forest models.

## License

This project is licensed under the MIT License.

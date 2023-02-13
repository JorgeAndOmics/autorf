import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def random_forest_model(dataset, target, n_trees=100, max_depth=None, min_samples_split=2, 
                        min_samples_leaf=1, max_features="auto", criterion="gini", 
                        sample_selection="auto"):
    """
    A function to implement a Random Forest approach using a single function with parameters as arguments. 
    The script would take in a dataset and a set of parameters, and use the Random Forest algorithm to train and 
    test a model on the dataset.
    
    Parameters:
    dataset (pd.DataFrame or np.array): The dataset, in a format such as a Pandas DataFrame or a NumPy array.
    target (str or int): The target variable, as a string or integer.
    n_trees (int, optional): The number of trees in the forest (default value = 100).
    max_depth (int or None, optional): The maximum depth of each tree (default value = None).
    min_samples_split (int, optional): The minimum number of samples required to split an internal node (default value = 2).
    min_samples_leaf (int, optional): The minimum number of samples required to be a leaf node (default value = 1).
    max_features (str or int, optional): The number of features to consider when looking for the best split (default value = "auto").
    criterion (str, optional): The criterion for measuring the quality of a split (default value = "gini").
    sample_selection (str, optional): The method for selecting samples for training each tree (default value = "auto").
    
    Returns:
    model (RandomForestClassifier): Trained Random Forest model.
    """
    
    # Convert the dataset to a Pandas DataFrame if it's a NumPy array
    if type(dataset) == np.ndarray:
        dataset = pd.DataFrame(dataset)
    
    # Split the dataset into features and target
    X = dataset.drop(target, axis=1)
    y = dataset[target]
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, 
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                   max_features=max_features, criterion=criterion, 
                                   bootstrap=sample_selection)
    model.fit(X, y)
    
    return model

def feature_importance_plot(model, X):
    """
    A function to plot the feature importances of the trained Random Forest model.
    
    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    X (pd.DataFrame): Features of the dataset.
    
    Returns:
    None
    """
    
    # Get the feature importances
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.show()

def decision_surface_plot(model, X, y):
    """
    A function to plot the decision surface of the trained Random Forest model.
    
    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    X (pd.DataFrame): Features of the dataset.
    y (pd.Series): Target of the dataset.
    
    Returns:
    None
    """
    
    # Get the decision surface
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision surface
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, alpha=0.8)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('Decision Surface Plot')
    plt.show()
    
def export_results(model, X, y, file_name):
    """
    A function to export the results of the analysis in a format that can be easily imported into other tools, 
    such as visualization software or other machine learning algorithms.
    
    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    X (pd.DataFrame): Features of the dataset.
    y (pd.Series): Target of the dataset.
    file_name (str): File name to save the results.
    
    Returns:
    None
    """
    
    # Save the results to a CSV file
    results = pd.concat([X, y], axis=1)
    results.to_csv(file_name, index=False)

def batch_process(datasets, targets, n_trees=100, max_depth=None, min_samples_split=2, 
                  min_samples_leaf=1, max_features="auto", criterion="gini", 
                  sample_selection="auto"):
    """
    A function to perform a batch process, in which multiple datasets can be processed at the same time.
    
    Parameters:
    datasets (list of pd.DataFrame or np.array): List of datasets.
    targets (list of str or int): List of target variables, as strings or integers.
    n_trees (int, optional): The number of trees in the forest (default value = 100).
    max_depth (int or None, optional): The maximum depth of each tree (default value = None).
    min_samples_split (int, optional): The minimum number of samples required to split an internal node (default value = 2).
    min_samples_leaf (int, optional): The minimum number of samples required to be a leaf node (default value = 1).
    max_features (str or int, optional): The number of features to consider when looking for the best split (default value = "auto").
    criterion (str, optional): The criterion for measuring the quality of a split (default value = "gini").
    sample_selection (str, optional): The method for selecting samples for training each tree (default value = "auto").
    
    Returns:
    models (list of RandomForestClassifier): List of trained Random Forest models.
    """
    
    models = []
    for dataset, target in zip(datasets, targets):
        models.append(random_forest_model(dataset, target, n_trees, max_depth, min_samples_split, 
                                           min_samples_leaf, max_features, criterion, sample_selection))
        
    return models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Model for Bioinformatics')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--target', type=str, required=True, help='Target variable name')
    parser.add_argument('--n_trees', type=int, default=100, help='Number of trees in the forest (default value = 100)')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of each tree (default value = None)')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum number of samples required to split an internal node (default value = 2)')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum number of samples required to be a leaf node (default value = 1)')
    parser.add_argument('--max_features', type=str, default="auto", help='Number of features to consider when looking for the best split (default value = "auto")')
    parser.add_argument('--criterion', type=str, default="gini", help='Criterion for measuring the quality of a split (default value = "gini")')
    parser.add_argument('--sample_selection', type=str, default="auto", help='Method for selecting samples for training each tree (default value = "auto")')
    args = parser.parse_args()
    
    # Load the dataset
    df = pd.read_csv(args.dataset)
    
    # Train and test the Random Forest model
    model = random_forest_model(df, args.target, args.n_trees, args.max_depth, args.min_samples_split, 
                                 args.min_samples_leaf, args.max_features, args.criterion, args.sample_selection)
    
    # Plot the feature importances
    feature_importance_plot(model, df.drop(args.target, axis=1))
    
    # Plot the decision surface (if the dataset has only two features)
    if df.shape[1] == 2:
        decision_surface_plot(model, df.drop(args.target, axis=1), df[args.target])
    
    # Export the results
    export_results(model, df.drop(args.target, axis=1), df[args.target], 'results.csv')


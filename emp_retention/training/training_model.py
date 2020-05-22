
import argparse
import os
from azureml.core import Run
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def generate_model(dataset, run):
    # Convert dataframe into numpy objects and split them into
    # train and test sets: 80/20
    X = dataset.loc[:, dataset.columns != "left"].values
    y = dataset.loc[:, dataset.columns == "left"].values.flatten()

    X_train, X_test, y_train, y_test
      = train_test_split(X, y, test_size=0.2, 
                         stratify=y, random_state=1)

    clf = LogisticRegression(solver='liblinear', random_state=0)
    clf.fit(X_train, y_train)

    # View the model's coefficients and bias
    run.log('Coefficients', clf.coef_)
    run.log('Bias', clf.intercept_)

    y_pred_LR = clf.predict(X_test)

    # Display confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred_LR)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1} {v2} ({v3})" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    run.log('Confusion Matrix', labels)

    # Display statistics
    accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    precision = cf_matrix[1, 1] / sum(cf_matrix[:, 1])
    recall = cf_matrix[1, 1] / sum(cf_matrix[1, :])
    f1_score = 2*precision*recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}" +
                     "\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                     accuracy, precision, recall, f1_score)
    run.log('Statistics', stats_text)

    # Log confusion matrix as JSON.
    cf_matrix_json = {"schema_type": "confusion_matrix",
                      "schema_version": "v1",
                      "data": {"class_labels": group_names,
                               "matrix": cf_matrix.tolist()}}

    run.log_confusion_matrix('Employee Retention Confusion Matrix',
                             cf_matrix_json,
                             description='Confusion matrix generated \
                             for the run')

    # Log the following metrics to the parent run so that
    # these are available for model evaluation later.
    run.parent.log('Accuracy', accuracy)
    run.parent.log('Precision', precision)
    run.parent.log('Recall', recall)
    run.parent.log('F1-score', f1_score)
    return clf


def main():
    # retrieve argument configured through script_params in estimator
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_folder', dest='clean_data_folder',
                        type=str,
                        help='folder path that stores the clean data file')
    parser.add_argument('--new_model_folder', dest='new_model_folder',
                        type=str,
                        help='output folder path that stores the new \
                        model .pkl file name')
    parser.add_argument('--clean_data_file', dest='clean_data_file',
                        type=str,
                        help='name of the clean data file')
    parser.add_argument('--new_model_file', dest='new_model_file',
                        type=str,
                        help='name of the new model .pkl file')
    args = parser.parse_args()

    # get the current run
    run = Run.get_context()

    # Read dataset
    training_dataset_file = args.clean_data_folder + args.clean_data_file
    dataset = pd.read_csv(training_dataset_file)
    run.log('Read training data from file', training_dataset_file)

    # Generate model
    clf = generate_model(dataset, run)

    # For "output" PipelineData, the folder must be created using
    # os.makedirs() first, then only can write files into the folder.
    os.makedirs(args.new_model_folder, exist_ok=True)

    # Pass model file to next step
    model_pkl_file = args.new_model_folder + args.new_model_file
    joblib.dump(value=clf, filename=model_pkl_file)

    run.log('Output path of new model', model_pkl_file)


if __name__ == '__main__':
    main()

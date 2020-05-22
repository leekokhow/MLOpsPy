
import argparse
import os
from azureml.core import Run
import pandas as pd


def prepare_dataset(dataset, run):
    # Rename sales feature into department
    dataset = dataset.rename(columns={"sales": "department"})

    # Map salary into integers
    salary_map = {"low": 0, "medium": 1, "high": 2}
    dataset["salary"] = dataset["salary"].map(salary_map)

    # Create dummy variables for department feature
    dataset = pd.get_dummies(dataset, columns=["department"],
                             drop_first=True)

    # Get number of positve and negative examples
    pos = dataset[dataset["left"] == 1].shape[0]
    neg = dataset[dataset["left"] == 0].shape[0]
    run.log('Positive', 'Positive examples = {}'.format(pos))
    run.log('Negative', 'Negative examples = {}'.format(neg))
    run.log('Proportion', 'Proportion of positive to negative \
    examples = {:.2f}%'.format((pos / neg) * 100))
    return dataset


def main():
    # retrieve argument configured through script_params in estimator
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_file', dest='raw_data_file',
                        type=str, help='input raw data in .csv file')
    parser.add_argument('--clean_data_folder', dest='clean_data_folder',
                        type=str, help='output folder path that stores \
                        the clean data file')
    parser.add_argument('--clean_data_file', dest='clean_data_file',
                        type=str, help='name of the clean data file')
    args = parser.parse_args()

    # get hold of the current run
    run = Run.get_context()

    # Read dataset
    dataset = pd.read_csv(args.raw_data_file)
    run.log('Read raw data file', args.raw_data_file)

    # Clean the dataset to use for model training
    dataset = prepare_dataset(dataset, run)

    # For "output" PipelineData, the folder must be created using
    # os.makedirs() first, then only can write files into the folder.
    os.makedirs(args.clean_data_folder, exist_ok=True)

    # Write the dataset into .csv file for the next step to process
    dataset.to_csv(args.clean_data_folder + args.clean_data_file, index=False)
    run.log('Output path of clean data', args.clean_data_folder
            + args.clean_data_file)


if __name__ == '__main__':
    main()

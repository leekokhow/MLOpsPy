
import argparse
from azureml.core import Run
from azureml.core.model import Model
from sklearn.externals import joblib
import shutil

def main():
    # Retrieve argument configured through script_params in estimator
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_model_folder', dest='new_model_folder', type=str, help='input folder path for reading the new model .pkl file')
    parser.add_argument('--new_model_file', dest='new_model_file', type=str, help='name of the model .pkl file')
    parser.add_argument("--model_name", dest='model_name', type=str, help="Name of the model to register into Azure ML Workspace")
    args = parser.parse_args()

    # Get the current run
    run = Run.get_context()
    
    # Adding metrics to tags so that these information can be used for model comparison purpose.
    metrics = ['Accuracy','Precision','Recall','F1-score']
    tags = {}
    for key in metrics:
        tags[key] = run.parent.get_metrics(key).get(key)           
        
    # Register the new model, note the metric values are stored in "tags".
    workspace = run.experiment.workspace
    model_pkl_file = args.new_model_folder + args.new_model_file
    model = Model.register(workspace=workspace,
                           model_name=args.model_name,
                           model_path=model_pkl_file,
                           tags=tags)
    run.log('Model registered', 'New model ' + model.name + ' version ' + str(model.version))
        
if __name__ == '__main__':
    main()
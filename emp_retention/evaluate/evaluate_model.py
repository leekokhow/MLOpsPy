
import argparse
from azureml.core import Run
from azureml.core.model import Model
from azureml.exceptions import WebserviceException


def main():
    # retrieve argument configured through script_params in estimator
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest='model_name', type=str,
                help="Name of the model to retrieve from Workspace")
    args = parser.parse_args()

    # Get the current run
    run = Run.get_context()

    # Get metrics from current model and compare with the metrics
    # of the new model. The metrics of the new model can be retrieved
    # from run.parent.get_metrics, which were created in training_model.py
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    current_metrics = {}
    new_metrics = {}

    try:
        workspace = run.experiment.workspace
        # Get latest model
        model = Model(workspace, args.model_name)

        for key in metrics:
            current_metrics[key] = float(model.tags.get(key))
            new_metrics[key] = run.parent.get_metrics(key).get(key)
            run.log(key, 'current(ver '
                         + str(model.version)
                         + ')='
                         + model.tags.get(key)
                         + ' new='
                         + str(run.parent.get_metrics(key).get(key))
                    )

    except WebserviceException as e:
        if('ModelNotFound' in e.message):
            model = None
        else:
            raise

    # Perform comparison. Just do a simple comparison:
    # If Accuracy improves, proceed next step to register model.
    if(model is not None):
        if(new_metrics['Accuracy'] >= current_metrics['Accuracy']):
            run.log("Result", "New Accuracy is as good as current, \
                will proceed to register new model.")
        else:
            run.log("Result", "New Accuracy is worse than current, \
                will not register model. Processing cancelled.")
            run.parent.cancel()
    else:
        run.log("Result", "This is the first model, will proceed \
            to register the model.")


if __name__ == '__main__':
    main()

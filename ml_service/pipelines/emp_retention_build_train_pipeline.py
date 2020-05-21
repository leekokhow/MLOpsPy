from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os

from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import EstimatorStep

def main():
    e = Env()
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("get_workspace:")
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(
        aml_workspace,
        e.compute_name,
        e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(
        aml_workspace, e.aml_env_name, create_new=e.rebuild_env)  #
    run_config = RunConfiguration()
    run_config.environment = environment

    if (e.datastore_name):
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501

    # Get dataset name
    dataset_name = e.dataset_name
	file_name = e.file_name

	# Get datastore name
	datatstore = Datastore.get(aml_workspace, datastore_name)

    # Check to see if dataset exists
    if (dataset_name not in aml_workspace.datasets):
        raise Exception("Could not find dataset at \"%s\" in Workspace." % dataset_name)
    else:
	    # Download a registered FileDataset (training-data.csv) from workspace to local folder
        dataset = Dataset.get_by_name(aml_workspace, name=dataset_name)
        dataset.download(target_path='.', overwrite=True)

        # Upload data file from local to datastore.
        datastore.upload_files([file_name], target_path=dataset_name, overwrite=True) 
	
			
    # Reference the data uploaded to blob storage using DataReference
    # Assign the datasource to input_data variable which will be used to pass to the pipeline step.
    raw_data_file = DataReference(datastore=datastore,
                                  data_reference_name="Raw_Data_File",
                                  path_on_datastore=dataset_name + '/' + file_name)

    # Create the PipelineParameter and PipelineData used for prepare data step.
    # clean_data_folder is used to store the clean data file.
    # Take note when create PipelineData, you don't need to specify the full folder path, 
    # just need the name of the subfolder to be created e.g. "clean_data_folder"
    clean_data_file = PipelineParameter(name="clean_data_file", default_value="/clean_data.csv")
    clean_data_folder = PipelineData("clean_data_folder", datastore=datastore)

    # raw_data_file is a Datareference and produce clean data to be used for model training.
    prepDataStep = PythonScriptStep(name="Prepare Data",
                                    source_directory=e.sources_directory_train,
                                    script_name=e.data_prep_script_path, 
                                    arguments=["--raw_data_file", raw_data_file, 
                                    "--clean_data_folder", clean_data_folder,
                                    "--clean_data_file", clean_data_file],
                                    inputs=[raw_data_file],
                                    outputs=[clean_data_folder],
                                    compute_target=aml_compute)
		
	print("Step Prepare Data created")
	
    # Create the PipelineParameter and PipelineData used for model training step.
    # new_model_folder is used to store the model .pkl file.
    new_model_file = PipelineParameter(name="new_model_file ", default_value='/'+e.model_name+'.pkl')
    new_model_folder = PipelineData("new_model_folder", datastore=datastore)

	# Create an Estimator (in this case we use the SKLearn estimator)
    est = SKLearn(source_directory=e.sources_directory_train,
                  entry_script=e.train_script_path,
                  conda_packages=['scikit-learn==0.20.3'],
                  compute_target=aml_compute)

    trainingStep = EstimatorStep(name="Model Training", 
                                 estimator=est,
                                 estimator_entry_script_arguments=[
								 "--clean_data_folder", clean_data_folder,
                                 "--new_model_folder", new_model_folder,
                                 "--clean_data_file", clean_data_file.default_value,
                                 "--new_model_file", new_model_file.default_value],
                                 runconfig_pipeline_params=None, 
                                 inputs=[clean_data_folder], 
                                 outputs=[new_model_folder], 
                                 compute_target=aml_compute)
			   
    print("Step Train created")

    # Create a PipelineParameter to pass the name of the model to be evaluated.
    model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)

    evaluateStep = PythonScriptStep(name="Evaluate Model",
                                    source_directory=e.sources_directory_train,
                                    script_name=e.evaluate_script_path, 
                                    arguments=["--model_name", model_name_param],
                                    compute_target=aml_compute)
		
    print("Step Evaluate created")

    registerStep = PythonScriptStep(name="Register Model",
                                    source_directory=e.sources_directory_train,
                                    script_name=e.register_script_path, 
                                    arguments=["--new_model_folder", new_model_folder,
                                               "--new_model_file", new_model_file,
                                               "--model_name", model_name],
                                    inputs=[new_model_folder],
                                    compute_target=aml_compute)
	
    print("Step Register created")
	
    # Check run_evaluation flag to include or exclude evaluation step.
    if ((e.run_evaluation).lower() == 'true'):
        print("Include evaluation step before register step.")
		# Chain the steps in sequence.
        trainingStep.run_after(prepDataStep)
        evaluateStep.run_after(trainingStep)
        registerStep.run_after(evaluateStep)
    else:
        print("Exclude evaluation step and directly run register step.")
		# Chain the steps in sequence.
        trainingStep.run_after(prepDataStep)
        registerStep.run_after(trainingStep)

    pipeline = Pipeline(workspace=aml_workspace, steps=[registerStep])
	pipeline.validate()
	print ("Pipeline is built")
	
    pipeline._set_experiment_name
    published_pipeline = pipeline.publish(
        name=e.pipeline_name,
        description="Predict Employee Retention Model training/retraining pipeline",
        version=e.build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()

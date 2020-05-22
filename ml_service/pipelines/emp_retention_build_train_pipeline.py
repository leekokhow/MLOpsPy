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
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("get_workspace:")
    print(aml_workspace)

    aml_compute = get_compute(
        aml_workspace,
        e.compute_name,
        e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    environment = get_environment(
        aml_workspace, e.aml_env_name, create_new=e.rebuild_env)  #
    run_config = RunConfiguration()
    run_config.environment = environment

    if (e.datastore_name):
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name

    dataset_name = e.dataset_name
    file_name = e.file_name
    datatstore = Datastore.get(aml_workspace, datastore_name)

    if (dataset_name not in aml_workspace.datasets):
        raise Exception("Could not find dataset at \"%s\" in Workspace." % dataset_name)
    else:
        dataset = Dataset.get_by_name(aml_workspace, name=dataset_name)
        dataset.download(target_path='.', overwrite=True)
        datastore.upload_files([file_name], target_path=dataset_name, overwrite=True) 

    raw_data_file = DataReference(datastore=datastore,
        data_reference_name="Raw_Data_File",
        path_on_datastore=dataset_name + '/' + file_name)

    clean_data_file = PipelineParameter(name="clean_data_file",
        default_value="/clean_data.csv")
    clean_data_folder = PipelineData("clean_data_folder",
        datastore=datastore)

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

    new_model_file = PipelineParameter(name="new_model_file ",
        default_value='/'+e.model_name+'.pkl')
    new_model_folder = PipelineData("new_model_folder",
        datastore=datastore)
    est = SKLearn(source_directory=e.sources_directory_train,
            entry_script=e.train_script_path,
            conda_packages=['scikit-learn==0.20.3'],
            compute_target=aml_compute)

    trainingStep = EstimatorStep(
        name="Model Training",
        estimator=est,
        estimator_entry_script_arguments=
            ["--clean_data_folder", clean_data_folder,
             "--new_model_folder", new_model_folder,
             "--clean_data_file", clean_data_file.default_value,
             "--new_model_file", new_model_file.default_value],
        runconfig_pipeline_params=None,
        inputs=[clean_data_folder],
        outputs=[new_model_folder],
        compute_target=aml_compute)

    print("Step Train created")

    model_name_param = PipelineParameter(name="model_name",
        default_value=e.model_name)

    evaluateStep = PythonScriptStep(
        name="Evaluate Model",
        source_directory=e.sources_directory_train,
        script_name=e.evaluate_script_path,
        arguments=["--model_name", model_name_param],
        compute_target=aml_compute)

    print("Step Evaluate created")

    registerStep = PythonScriptStep(
        name="Register Model",
        source_directory=e.sources_directory_train,
        script_name=e.register_script_path,
        arguments=["--new_model_folder", new_model_folder,
            "--new_model_file", new_model_file,
            "--model_name", model_name_param],
        inputs=[new_model_folder],
        compute_target=aml_compute)

    print("Step Register created")

    if ((e.run_evaluation).lower() == 'true'):
        print("Include evaluation step before register step.")
        trainingStep.run_after(prepDataStep)
        evaluateStep.run_after(trainingStep)
        registerStep.run_after(evaluateStep)
    else:
        print("Exclude evaluation step and directly run register step.")
        trainingStep.run_after(prepDataStep)
        registerStep.run_after(trainingStep)

    pipeline = Pipeline(workspace=aml_workspace, steps=[registerStep])
    pipeline.validate()
    print("Pipeline is built")

    pipeline._set_experiment_name
    published_pipeline = pipeline.publish(
        name=e.pipeline_name,
        description="Predict Employee Retention Model training pipeline",
        version=e.build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()

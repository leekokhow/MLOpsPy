# MLOps with Azure ML

This example is based on [predict-employee-retention-part3-pipelines](https://github.com/leekokhow/azureml).

The purpose is to demonstrate how to automate an end to end ML workflow using Azure DevOps.

You may want to read this article by Ben Keen that covers [Creating End-to-End MLOps pipelines using Azure ML and Azure Pipelines](https://benalexkeen.com/creating-end-to-end-mlops-pipelines-using-azure-ml-and-azure-pipelines-part-1/)


### References

- Read [MLOps with Azure ML](https://github.com/microsoft/MLOpsPython) before you proceed.
- Read [Bring your own code with the MLOpsPython repository template](https://github.com/microsoft/MLOpsPython/blob/master/docs/custom_model.md) to create your own custom model (which is what I did for this example).

### Setup steps for this example

1. Create an Azure DevOps account, install the Machine Learning (by Microsoft DevLabs) extension.


2. Create an Azure DevOps project and named it as "MLOpsPy" and import repository from https://github.com/leekokhow/MLOpsPy. Launch this project and proceed the following steps.


3. [Create the variable group](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-a-variable-group-for-your-pipeline) called "devopsforai-aml-vg". Take note to change BASE_NAME to a unique name. You can refer to "Create Variables for MLOpsPy.xlsx" for the variables.


4. Create a resource connection in Azure DevOps:

Project Settings > Service connections > New service connection > Azure Resource Manager

Choose "Service principal (automatic)"

Click **Next**

Scope level : Subscription

Subscription : [choose your subscription]

[Resource group : mlopspy-RG (choose this if it has been created)]

Service connection name : azure-resource-connection

Checked "Grant access permission to all pipelines"

Cick **Save**


5. (Skip this if you have already created Azure Machine Learning service in the portal)

Run the ARM template to create the cloud resources:

Pipelines > New Pipeline > Azure Repos Git > MLOpsPy > Existing Azure Pipelines YAML file

Select Path: /environment_setup/iac-create-environment-pipeline-arm.yml
Click **Continue**
Click **Run**

Note: if you want to remove the environment, select path: /environment_setup/iac-create-environment-pipeline-arm.yml. 


6. Once Azure Machine Learning Workspace has been created successfully, create a "Compute clusters" to be used for model training:

Launch your Azure Machine Learning Studio > Compute > Compute clusters > +New

Location : Southeast Asia

Virtual machine priority : Dedicated

Virtual machine type : CPU

Virtual machine size : Select from recommended options

Choose "Standard_DS2_V2"

Click **Next**

Computer name: cpucluster

Minimum number of nodes : 0

Maximum number of nodes : 2

Idle seconds before scale down : 120

Click **Create**


7. To deploy the model into Azure Kubernetes, you need to create an "Inference clusters":

Launch your Azure Machine Learning Studio > Compute > Inference clusters > +New

Kubernetes Service : Create new

Location : Southeast Asia

Choose "Standard_D4_v2" (8 cores, 28GB RAM, 400GB storage)

Click **Next**

Compute name : aks

Cluster purpose : Dev-test

Click **Create**


8. Create a training dataset that serves as input for model training:
Launch your Azure Machine Learning Studio > Datasets > Registered datasets > +Create dataset > From local files

Name: predict-employee-retention-training-data

Dataset type: File

Click **Next**

Select or create a datastore: workspaceblobstore

Upload: Upload Files: <use the training-data.csv found in MLOpsPy/data/ folder>

Upload path : predict-employee-retention-training-data

Click **Next**

Click **Create**


9. Create a workspace connection in Azure DevOps:

Project Settings > Service connections > New service connection > Azure Resource Manager

Service Principal (automatic)

Click **Next**

Scope level : Machine Learning Workspace

Subscription : choose your subscription

Resource group : mlopspy-RG

Machine Learning Workspace : mlopspy-AML-WS

Service connection name : aml-workspace-connection

Checked "Grant access permission to all pipelines"

Click **Save**


10. Run the DevOps pipeline:
Pipelines > New Pipeline > Azure Repos Git > MLOpsPy > Existing Azure Pipelines YAML file

Select Path: /.pipelines/emp_retention-ci.yml

Click **Continue**

Click **Run**


11. Once the DevOps pipeline completed successfully, you can check for the ACI and AKS endpoints in Azure Machine Learning Studio.

Note: To avoid excessive cost, delete the Compute clusters, Inference clusters, models, endpoints that were created after your finish exploring this project.  

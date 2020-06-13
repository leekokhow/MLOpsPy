# MLOps with Azure ML

This example is based on [predict-employee-retention-part3-pipelines](https://github.com/leekokhow/azureml).

The purpose is to demonstrate how to automate an end to end ML workflow using Azure DevOps

### References

- Read [MLOps with Azure ML](https://github.com/microsoft/MLOpsPython) before you proceed.
- Read [Bring your own code with the MLOpsPython repository template](https://github.com/microsoft/MLOpsPython/blob/master/docs/custom_model.md) to create your own custom model (which is what I did for this example). 

### Setup steps for this example

1. Create an Azure DevOps project and named it as "MLOpsPy", launch this project and proceed the following steps.

2. Create the variables in "devopsforai-aml-vg" group. Take note to change BASE_NAME to a unique name.

3. Create resource connection:
Project Settings > Service Connection > Create Service connection > Azure Resource Manager

Service Principal (automatic)

Click **Next**

Scope level : Subscription

Subscription : [choose your subscription]

Service connection name : azure-resource-connection

Checked "Grant access permission to all pipelines"

Cick **Save**

4. Run the ARM template to create the cloud resources:
Pipelines > Create Pipeline > Azure Repos Git > MLOpsPy > Existing Azure Pipelines YAML file

Select Path: /environment_setup/iac-create-environment-pipeline-arm.yml

5. Create Compute clusters in AML Workspace:
Computer name: cpucluster

Virtual machine size STANDARD_DS2_V2

Virtual machine priority Dedicated

Minimum number of nodes 0

Maximum number of nodes 2

Idle seconds before scale down 120

6. Run pipeline "/.pipelines/emp_retention-ci.yml".

7. Once your workspace has been created in Azure, create a workspace connection in Azure DevOps:
Project Settings > Service Connection > New Service connection > Azure Resource Manager

Service Principal (automatic)

Click **Next**

Scope level : Machine Learning Workspace

Subscription : choose your subscription

Resource group : mlopspy-RG

Machine Learning Workspace : mlopspy-AML-WS

Service connection name : aml-workspace-connection

Checked "Grant access permission to all pipelines"

Click **Save**

8. Use Azure Kubernetes
- You need to create "Inference Cluster" first: 

Option 1: Create a new AKS cluster : Name it as "aks" with 2 x Standard D4 v2 (8 vcpus, 28 GiB memory) for Dev-test environment. Requires minimum of 12 vcpus.

Option 2: Refer to [predict-employee-retention-part3-pipelines](https://github.com/leekokhow/azureml/blob/master/predict-employee-retention-part3-pipelines.ipynb) section K.1 on how to create a DEV_TEST AKS cluster. 

- Make sure these variables are added to devopsforai-aml-vg group in Azure DevOps:

AKS_COMPUTE_NAME : aks

AKS_DEPLOYMENT_NAME : mlops-aks

9. Run "/.pipelines/emp_retention-ci.yml" to start the DevOps pipeline. 

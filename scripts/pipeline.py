import json

from azureml.core import Dataset, Environment, Experiment, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Open JSON file containing Service Principal credentials
# returns JSON object as a dictionary
with open('./auth.json') as file:
    authentication_credentials = json.load(file)

# Create Service Principal Authentication object
sp = ServicePrincipalAuthentication(
    tenant_id=authentication_credentials['tenant-id'],
    service_principal_id=authentication_credentials['principal-id'],
    service_principal_password=authentication_credentials['principal-secret']
)

# Load workspace
ws = Workspace.from_config(auth=sp)

# Load environment
env = Environment('motor-fault-classification')

# Create compute target
compute_name = 'azureml-motor-fault'
training_instance = ComputeTarget(workspace=ws, name=compute_name)

# Get a dataset for the initial data
file_ds_normal = Dataset.get_by_name(ws, 'normal')
# file_ds_horizontal = Dataset.get_by_name(ws, 'horizontal')
# file_ds_vertical = Dataset.get_by_name(ws, 'vertical')
# file_ds_imbalance = Dataset.get_by_name(ws, 'imbalance')
# file_ds_overhang = Dataset.get_by_name(ws, 'overhang')
# file_ds_underhang = Dataset.get_by_name(ws, 'underhang')

# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = OutputFileDatasetConfig('prepped')

# Step to run a Python script
step1 = PythonScriptStep(
    name='prepare data',
    source_directory='scripts',
    script_name='data_prep.py',
    compute_target=training_instance,
    arguments=[
        '--start-index', 0,
        '--resample-rate', 100,
        '--out-folder', prepped_data,
        '--ds-normal', file_ds_normal.as_download()
        # '--ds-horizontal', file_ds_horizontal.as_download(),
        # '--ds-vertical', file_ds_vertical.as_download(),
        # '--ds-imbalance', file_ds_imbalance.as_download(),
        # '--ds-overhang', file_ds_overhang.as_download(),
        # '--ds-underhang', file_ds_underhang.as_download()
    ]
)

# TODO: Step to run modelling

# Construct the pipeline
train_pipeline = Pipeline(workspace=ws, steps=[step1])

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name='training-pipeline')
pipeline_run = experiment.submit(train_pipeline)

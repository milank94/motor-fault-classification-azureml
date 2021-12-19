import json

from azureml.core import Environment, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

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

# Create environment
env = Environment.from_conda_specification(
    name='training_environment',
    file_path='./environment.yml'
)

# Register environment
env.register(workspace=ws)

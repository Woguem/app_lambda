import boto3

# Initialiser le client SageMaker Runtime
runtime = boto3.client("sagemaker-runtime", region_name="eu-west-3")

# Données d'entrée au format CSV (une ligne = une instance)
payload = "15.1,12.5,17.4,15.2"

# Appel à l’endpoint
response = runtime.invoke_endpoint(
    EndpointName="my-real-time-endpoint",
    ContentType="text/csv",
    Body=payload
)

# Résultat
print("Model response :", response["Body"].read().decode("utf-8"))

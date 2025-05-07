# lambda_function.py
import boto3
import json
from datetime import datetime
import pandas as pd
import io
from imblearn.over_sampling import SMOTE
from time import sleep
from sagemaker import image_uris


def balance_data(data):
    """Balance the data using SMOTE or other techniques."""
    
    # Separate features and target variable
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Appy SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Recreate the DataFrame with balanced data
    balanced_data = pd.DataFrame(X_res, columns=X.columns)
    balanced_data['target'] = y_res
    
    print(f"Balanced class distribution: {balanced_data['target'].value_counts()}")
    
    return balanced_data


# Get the URI for the built-in XGBoost image
image_uri = image_uris.retrieve(
    framework='xgboost',
    region='eu-west-3',
    version='1.5-1'
)



def lambda_handler(event, context):
    # Initialize clients
    sagemaker = boto3.client('sagemaker')
    s3 = boto3.client('s3')
    
    # Get bucket and key from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Key of existing data in S3
    base_key = 'data.csv'
    
    # Download existing data from S3
    try:
        base_obj = s3.get_object(Bucket=bucket, Key=base_key)
        base_df = pd.read_csv(base_obj['Body'])
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                "error": str(e)})
        }  
        
    # Download new data from S3
    new_obj = s3.get_object(Bucket=bucket, Key=key)
    new_df = pd.read_csv(new_obj['Body'], sep=",")

    # Combine the dataframes
    df = pd.concat([base_df, new_df], ignore_index=True)
    
    data_balanced = balance_data(df)
    
    # Save the balanced dataset back to S3
    balanced_key = 'balanced_data.csv'
    
    try:
        # Convert DataFrame to CSV and upload to S3
        columns = [col for col in data_balanced.columns if col != 'target'] + ['target']
        data_balanced = data_balanced[columns]
        data_balanced['target'] = data_balanced['target'].astype(int)
        print(data_balanced['target'].unique())

        data_balanced = data_balanced.dropna()
        csv_buffer = io.StringIO()
        data_balanced.to_csv(csv_buffer, index=False, header=False)
        s3.put_object(Bucket=bucket, Key=balanced_key, Body=csv_buffer.getvalue())
        print(f"Balanced data saved to S3 at {balanced_key}")
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                "error": f"Error saving balanced data to S3: {str(e)}"})
        }
    
    # Create training job name
    job_name = f"training-job-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
# Start training job
    response = sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File'
        },
        RoleArn='arn:aws:iam::556417283723:role/accesss3',
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{bucket}/{balanced_key}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv',   
            }
        ],
        OutputDataConfig={
            'S3OutputPath': f's3://{bucket}/training_model/'
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        },
        HyperParameters={ 
            "max_depth": "5",
            "eta": "0.2",
            "subsample": "0.8",
            "objective": "multi:softmax",
            "num_class": "3",
            "num_round": "100",
            "eval_metric": "mlogloss"
        },
    )
    
    # Wait for training job to complete
    try:
        print(f"Waiting for training job {job_name} to complete...")
        waiter = sagemaker.get_waiter('training_job_completed_or_stopped')
        waiter.wait(TrainingJobName=job_name)
        
        # Describe the training job to get the model artifacts
        training_job_info = sagemaker.describe_training_job(TrainingJobName=job_name)
        model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']
        
        # Create model name
        model_name = f"model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create model in SageMaker
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_artifacts
            },
            ExecutionRoleArn='arn:aws:iam::556417283723:role/accesss3'
        )
        
        # Create endpoint config name
        endpoint_config_name = f"endpoint-config-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create endpoint configuration
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium'
                }
            ]
        )
        
        # Create or update endpoint
        endpoint_name = "my-real-time-endpoint"
        
        try:
            # Check if endpoint exists
            sagemaker.describe_endpoint(EndpointName=endpoint_name)
            # If exists, update it
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"Updated endpoint {endpoint_name} with new model")
        except:
            # If doesn't exist, create it
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"Created new endpoint {endpoint_name}")
        
        # Enhanced endpoint deployment waiting with timeout
        print(f"Waiting for endpoint {endpoint_name} to be in service...")
        max_retries = 60  # 60 * 30 seconds = 30 minutes max
        current_retry = 0
        endpoint_ready = False
        
        while current_retry < max_retries:
            endpoint_status = sagemaker.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
            if endpoint_status == 'InService':
                endpoint_ready = True
                break
            elif endpoint_status in ['Failed', 'RollbackFailed', 'RollbackComplete']:
                raise Exception(f"Endpoint deployment failed with status: {endpoint_status}")
            
            print(f"Endpoint status: {endpoint_status} - Waiting... (attempt {current_retry + 1}/{max_retries})")
            sleep(30)  # Wait 30 seconds between checks
            current_retry += 1
        
        if not endpoint_ready:
            raise Exception("Endpoint deployment timed out after 30 minutes")
        
        print(f"Endpoint {endpoint_name} is now InService")
        
        # Perform a sample inference
        runtime = boto3.client('runtime.sagemaker')
        
        # Get a sample row from the data (without target)
        sample_data = data_balanced.drop(columns=['target']).iloc[0:1].values.tolist()[0]
        
        # Convert to CSV string
        payload = ','.join(map(str, sample_data))
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=payload
        )
        
        result = response['Body'].read().decode()
        print(f"Sample inference result: {result}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": "Training, deployment and inference completed successfully",
                "training_job": job_name,
                "model": model_name,
                "endpoint": endpoint_name,
                "sample_inference_result": result
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                "error": f"Error in training/deployment process: {str(e)}",
                "training_job": job_name
            })
        }

    


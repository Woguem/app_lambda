# lambda_function.py
import boto3
import json
from datetime import datetime
import pandas as pd
import io
from imblearn.over_sampling import SMOTE

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
    new_df = pd.read_csv(new_obj['Body'])

    # Combine the dataframes
    df = pd.concat([base_df, new_df], ignore_index=True)
    
    data_balanced = balance_data(df)
    
    # Save the balanced dataset back to S3
    balanced_key = 'balanced_data.csv'
    
    try:
        # Convert DataFrame to CSV and upload to S3
        csv_buffer = io.StringIO()
        data_balanced.to_csv(csv_buffer, index=False)
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
                }
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
            "objective": "binary:logistic",
            "num_round": "100"
        },
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Started training job: {job_name}')
    }


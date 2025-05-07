Lambda-triggered XGBoost Training Pipeline
This repository contains a Dockerized AWS Lambda function that automates machine learning model training and deployment using XGBoost.

🔁 Workflow Overview

1- Trigger: A new file upload to an S3 bucket triggers the Lambda function.

2- Data Handling: The new data is concatenated with an existing dataset stored in S3.

3- Class Balancing: The combined dataset undergoes class balancing to handle imbalanced distributions.

4- Model Training: An XGBoost model is trained on the balanced dataset.

5- Deployment: The trained model is deployed for real-time inference.

6- Inference: Future predictions can be served via the deployed endpoint.

🐳 Docker
The Lambda function runs inside a Docker container, enabling custom dependencies and runtime environments.

📁 Structure
    1- Dockerfile: Builds the Lambda container.

    2- lambda_function.py: Main logic for data prep, training, and deployment.

🚀 Usage
Push a CSV file to the configured S3 bucket to automatically trigger training.
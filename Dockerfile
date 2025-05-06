# Utilise l'image de base fournie par AWS pour Lambda Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# Copy requirements.txt file
COPY ./requirements.txt .


# Install  dependencies (add gccc++ for imbalanced-learn) 
RUN yum install -y gcc-c++ python3-devel atlas-sse3-devel && \
    pip install --upgrade pip && \
    pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" && \
    yum clean all

# Copy the rest of the application code
COPY ./ ${LAMBDA_TASK_ROOT}

# Set the command to run the Lambda function
CMD ["sagemakertrainer.lambda_handler"]

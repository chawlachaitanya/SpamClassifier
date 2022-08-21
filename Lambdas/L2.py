import boto3
import json


def lambda_handler(event, context):
    client = boto3.client('sagemaker')
    client.start_notebook_instance(NotebookInstanceName='spamclassifier')
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

# AWS Lambda Product Recommendation System

This project implements a product recommendation system using AWS Lambda, AWS Bedrock, and OpenSearch. It uses RAG (Retrieval Augmented Generation) to provide personalized product recommendations based on user queries.

## Architecture

- AWS Lambda: Serverless compute for handling recommendation requests
- AWS Bedrock: For generating embeddings and recommendations using Claude
- OpenSearch: Vector database for storing and retrieving product information
- LangChain: Framework for building the RAG pipeline

## Prerequisites

1. AWS Account with access to:
   - AWS Lambda
   - AWS Bedrock
   - OpenSearch Service
2. Python 3.8 or later
3. AWS CLI configured with appropriate credentials

## Setup

1. Create an OpenSearch domain in AWS:
   - Enable fine-grained access control
   - Configure security groups and VPC settings
   - Note down the endpoint URL

2. Create an AWS Lambda function:
   - Runtime: Python 3.8
   - Memory: 512 MB (minimum)
   - Timeout: 30 seconds
   - Environment variables:
     - `OPENSEARCH_ENDPOINT`: Your OpenSearch domain endpoint
     - `OPENSEARCH_INDEX`: Index name (default: 'product-recommendations')
     - `AWS_REGION`: Your AWS region

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Deploy the Lambda function:
   - Create a deployment package with all dependencies
   - Upload to AWS Lambda

5. Set up IAM permissions:
   - Add permissions for Bedrock and OpenSearch access
   - Required permissions:
     - `bedrock:InvokeModel`
     - `es:ESHttp*`

## Usage

1. Index your product data:
   ```bash
   python index_products.py
   ```

2. Invoke the Lambda function with a query:
   ```json
   {
     "query": "I'm looking for wireless headphones with good battery life"
   }
   ```

3. The function will return personalized recommendations based on the query.

## Response Format

```json
{
  "statusCode": 200,
  "body": {
    "recommendations": "Detailed recommendations based on the query...",
    "query": "Original query"
  }
}
```

## Error Handling

The Lambda function includes error handling for:
- Missing query parameters
- OpenSearch connection issues
- Bedrock API errors

## Security Considerations

1. Ensure proper IAM roles and permissions
2. Use VPC endpoints for OpenSearch
3. Enable encryption at rest and in transit
4. Implement proper authentication and authorization

## Monitoring and Logging

- Use CloudWatch Logs to monitor Lambda execution
- Set up CloudWatch Alarms for error rates and latency
- Monitor OpenSearch cluster health

## Contributing

Feel free to submit issues and enhancement requests! 
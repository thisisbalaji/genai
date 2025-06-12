import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.embeddings import BedrockEmbeddings
import os

def get_opensearch_client():
    """Initialize OpenSearch client"""
    region = os.environ.get('AWS_REGION', 'us-east-1')
    opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
    
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'es',
        session_token=credentials.token
    )
    
    return OpenSearch(
        hosts=[{'host': opensearch_endpoint, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

def get_embeddings():
    """Initialize Bedrock embeddings"""
    bedrock = boto3.client('bedrock-runtime')
    return BedrockEmbeddings(
        client=bedrock,
        model_id="amazon.titan-embed-text-v1"
    )

def create_index(client, index_name):
    """Create OpenSearch index with vector search mapping"""
    mapping = {
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "name": {"type": "text"},
                "description": {"type": "text"},
                "category": {"type": "keyword"},
                "price": {"type": "float"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib"
                    }
                }
            }
        }
    }
    
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=mapping)

def index_product(client, index_name, product_data, embeddings):
    """Index a single product with its vector embedding"""
    # Generate embedding for product description
    description_embedding = embeddings.embed_query(product_data['description'])
    
    # Prepare document with vector
    document = {
        "product_id": product_data['id'],
        "name": product_data['name'],
        "description": product_data['description'],
        "category": product_data['category'],
        "price": product_data['price'],
        "vector": description_embedding
    }
    
    # Index the document
    client.index(
        index=index_name,
        id=product_data['id'],
        body=document
    )

def main():
    # Initialize clients
    opensearch_client = get_opensearch_client()
    embeddings = get_embeddings()
    
    # Create index
    index_name = os.environ.get('OPENSEARCH_INDEX', 'product-recommendations')
    create_index(opensearch_client, index_name)
    
    # Example product data - replace with your actual product data
    products = [
        {
            "id": "1",
            "name": "Wireless Headphones",
            "description": "Premium noise-cancelling wireless headphones with 30-hour battery life",
            "category": "Electronics",
            "price": 199.99
        },
        # Add more products here
    ]
    
    # Index products
    for product in products:
        index_product(opensearch_client, index_name, product, embeddings)
        print(f"Indexed product: {product['name']}")

if __name__ == "__main__":
    main() 
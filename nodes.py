import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from requests_aws4auth import AWS4Auth
from typing import Dict, List, Any, TypedDict
import os

class AgentState(TypedDict):
    query: str
    product_context: List[str]
    recommendations: str
    error: str

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
region = os.environ.get('AWS_REGION', 'us-east-1')

# OpenSearch configuration
opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
opensearch_index = os.environ.get('OPENSEARCH_INDEX', 'product-recommendations')

# Initialize OpenSearch client
credentials = boto3.Session().get_credentials()
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    'es',
    session_token=credentials.token
)

opensearch_client = OpenSearch(
    hosts=[{'host': opensearch_endpoint, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

def get_bedrock_llm():
    """Initialize Bedrock LLM"""
    return Bedrock(
        client=bedrock,
        model_id="anthropic.claude-v2",
        model_kwargs={"temperature": 0.7, "max_tokens": 1000}
    )

def get_embeddings():
    """Initialize Bedrock embeddings"""
    return BedrockEmbeddings(
        client=bedrock,
        model_id="amazon.titan-embed-text-v1"
    )

def search_products(state: AgentState) -> AgentState:
    """Search for products using OpenSearch"""
    try:
        vector_store = OpenSearchVectorSearch(
            index_name=opensearch_index,
            embedding_function=get_embeddings(),
            opensearch_url=f"https://{opensearch_endpoint}",
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            vector_size=1536
        )
        
        results = vector_store.similarity_search(state["query"], k=5)
        state["product_context"] = [doc.page_content for doc in results]
        return state
    except Exception as e:
        state["error"] = f"Error searching products: {str(e)}"
        return state

def generate_recommendations(state: AgentState) -> AgentState:
    """Generate product recommendations using Bedrock"""
    try:
        llm = get_bedrock_llm()
        
        prompt = f"""Based on the following user query and product information, provide personalized product recommendations:

User Query: {state['query']}

Product Information:
{chr(10).join(state['product_context'])}

Please provide detailed recommendations explaining why these products would be suitable for the user."""

        state["recommendations"] = llm.predict(prompt)
        return state
    except Exception as e:
        state["error"] = f"Error generating recommendations: {str(e)}"
        return state

def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing or end"""
    if state.get("error"):
        return "end"
    return "continue" 
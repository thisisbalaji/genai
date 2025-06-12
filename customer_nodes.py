
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from typing import Dict

# Connect to OpenSearch vector store
embedding = BedrockEmbeddings()
vectorstore = OpenSearchVectorSearch(
    index_name="customer-profiles",
    embedding_function=embedding,
    opensearch_url="https://your-opensearch-endpoint",
    http_auth=("admin", "admin")  # Use secure credentials in real deployment
)

# Node: Retrieve customer profile
def customer_profile_node(state: Dict) -> Dict:
    query = state.get("query", "preferences of customer John Doe")
    docs = vectorstore.similarity_search(query, k=3)
    state["documents"] = [doc.page_content for doc in docs]
    return state

# Node: Summarize profile with Claude 3 via Bedrock
def summarize_profile_node(state: Dict) -> Dict:
    llm = Bedrock(model_id="anthropic.claude-3-sonnet-20240229", region_name="us-east-1")
    docs = "\n".join(state.get("documents", []))
    prompt = f"Summarize the following customer profile in 3 bullet points:\n{docs}"
    state["summary"] = llm.invoke(prompt)
    return state

# Node: Generate mock recommendations
def recommendation_node(state: Dict) -> Dict:
    state["recommendations"] = [
        "Nike Running Shoes",
        "Under Armour T-shirt",
        "Apple Watch Series 9"
    ]
    return state

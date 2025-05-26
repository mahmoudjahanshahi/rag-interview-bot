# %%
import os
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey

# Set up environment variables for Azure OpenAI
oai_client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'), 
    api_version = os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
)

# Set up environment variables for Azure OpenAI Embeddings
emb_client = AzureOpenAI(
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
    api_version = os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint = os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
)

# Set up environment variables for Azure Cosmos DB
url = os.getenv('COSMOS_DB_ENDPOINT')
key = os.getenv('COSMOS_DB_KEY')
db_client = CosmosClient(url, credential=key)

# %%
# Function to create embeddings for text chunks
def create_embeddings(text, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")):
    # Create embeddings for each document chunk
    embeddings = emb_client.embeddings.create(input = text, model=model).data[0].embedding
    return embeddings

# %%
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# Load the data from Azure Cosmos DB
container = db_client.get_database_client('interview-assistant').get_container_client('chunks')
items = list(container.read_all_items())
flattened_df = pd.DataFrame(items)

# Create the search index
X = np.vstack(flattened_df['embedding'].to_numpy())
nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(X)

# %%
# Function to handle user input and generate a response
def chatbot(user_input, neighbors=8):
    """
    Handles user input, retrieves relevant documents, and generates a response using Azure OpenAI.
    Args:
        user_input (str): The user's question or input.
    Returns:
        str: The generated response from the AI assistant.
    """
    # Convert the question to a query vector
    query_vector = create_embeddings(user_input)

    # Find the most similar documents
    _, indices = nbrs.kneighbors([query_vector], n_neighbors=neighbors)

    # Use a set to avoid duplicates
    indices_set = set(indices[0])  

    # Retrieve text chunks
    context_chunks = [
        f"[SOURCE: {flattened_df['source'].iloc[i]}]\n{flattened_df['chunk'].iloc[i]}"
        for i in indices_set
    ]

    # Combine context and user question
    context_text = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context_text}\n\nBased only on the provided context, answer this question: {user_input}"

    # Create message payload
    messages = [
        {"role": "system", "content": "You are an AI assistant helping a candidate prepare for job interviews. Only use the provided context (resume, job description, interview tips, company info) to answer. Do not make up facts or experiences. If the answer cannot be inferred from context, respond with 'I dont have enough information to answer that.'. Try to be brief."},
        {"role": "user", "content": prompt}
    ]

    # use chat completion to generate a response
    response = oai_client.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        temperature=0.3,
        max_tokens=200,
        messages=messages
    )

    return response.choices[0].message.content.strip()

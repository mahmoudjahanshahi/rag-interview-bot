# %%
# pip install --upgrade pip
# pip install openai
# pip install azure-cosmos

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
# Create database if it doesn't exist
database = db_client.create_database_if_not_exists(id='interview-assistant')

# Create container if it doesn't exist
container = database.create_container_if_not_exists(
    id='chunks',
    partition_key=PartitionKey(path='/source')
)

# %%
import pandas as pd

# Specify the paths to text files
data_paths = ['../data/company.txt', '../data/job_description.txt', '../data/interview_tips.txt', '../data/resume.txt']

# Read the text files and create a DataFrame
rows = []

for path in data_paths:
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        rows.append({'path': path, 'text': file_content})

df = pd.DataFrame(rows)

# %%
import re

# Function to split text into chunks based on word count
def split_text_with_overlap(text, max_words=100, min_words=40, overlap=20):
    """
    Splits text into sentence-based chunks with overlapping word windows.
    
    Args:
        text (str): The input text.
        max_words (int): Max words per chunk.
        min_words (int): Min words per chunk (ignored for final chunk).
        overlap (int): Number of words to overlap between chunks.
        
    Returns:
        list: List of overlapping sentence-based text chunks.
    """

    # Split by sentence boundaries: punctuation and newlines
    sentence_endings = re.compile(r'(?<=[.!?])\s+|\n+')
    sentences = sentence_endings.split(text.strip())

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words_in_sentence = sentence.split()
        current_chunk.extend(words_in_sentence)
        word_count += len(words_in_sentence)

        if word_count >= max_words:
            # Finalize current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Create overlap for next chunk
            current_chunk = current_chunk[-overlap:]
            word_count = len(current_chunk)

    # Handle final chunk
    if current_chunk:
        if len(current_chunk) >= min_words or not chunks:
            chunks.append(' '.join(current_chunk))
        else:
            # Append to previous if too small
            chunks[-1] += ' ' + ' '.join(current_chunk)

    return chunks

# %%
# Split the text in the DataFrame into chunks
splitted_df = df.copy()
splitted_df['chunks'] = splitted_df['text'].apply(lambda x: split_text_with_overlap(x))

# Flatten the DataFrame to have one chunk per row
flattened_df = splitted_df.explode('chunks')
flattened_df = flattened_df.reset_index(drop=True)

# %%
# Function to create embeddings for text chunks
def create_embeddings(text, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")):
    # Create embeddings for each document chunk
    embeddings = emb_client.embeddings.create(input = text, model=model).data[0].embedding
    return embeddings

# %%
# create embeddings for the whole data chunks and store them in a list
embeddings = []
for chunk in flattened_df['chunks']:
    embeddings.append(create_embeddings(chunk))

# store the embeddings in the dataframe
flattened_df['embeddings'] = embeddings

# Upsert the data into Azure Cosmos DB
for idx, row in flattened_df.iterrows():
    container.upsert_item({
        "id": str(idx),
        "source": os.path.basename(row['path']),
        "chunk": row['chunks'],
        "embedding": row['embeddings']
    }
    )

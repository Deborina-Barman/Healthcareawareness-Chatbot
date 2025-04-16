import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_urls_from_file,
    fetch_data_from_urls,
    split_text_chunks,
    download_hugging_face_embeddings
)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("🛑 PINECONE_API_KEY not found in environment variables.")

# Step 1: Load URLs from your JSON file
url_file = "data/urls.json"
urls = load_urls_from_file(url_file)

# Step 2: Load and process content
documents = fetch_data_from_urls(urls)
text_chunks = split_text_chunks(documents)

# Step 3: Load HuggingFace Embeddings
embeddings = download_hugging_face_embeddings("BAAI/bge-small-en-v1.5")

# Step 4: Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "healthcare-awareness-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 5: Upload to Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"✅ Successfully indexed {len(text_chunks)} chunks to Pinecone index: '{index_name}'")

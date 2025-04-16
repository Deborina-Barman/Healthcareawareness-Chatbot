import os
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Configure logging to show timestamp and messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')


# ✅ Utility function to validate whether a string is a proper URL
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


# ✅ Load URLs from either a .txt file or a .json file
def load_urls_from_file(file_path: str) -> list:
    ext = Path(file_path).suffix.lower()  # Get file extension
    urls = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # If file is JSON, load URLs from key "urls"
            if ext == ".json":
                data = json.load(f)
                urls = data.get("urls", []) if isinstance(data, dict) else data
            # If file is TXT, read one URL per line
            elif ext == ".txt":
                urls = [line.strip() for line in f if line.strip()]
            else:
                logging.warning("Unsupported file format. Please use .txt or .json")
                return []

        # Filter out any invalid URLs
        valid_urls = [url for url in urls if is_valid_url(url)]
        logging.info(f"Loaded {len(valid_urls)} valid URLs from {file_path}")
        return valid_urls

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return []


# ✅ Fetch HTML/text content from the list of valid URLs
def fetch_data_from_urls(urls: list):
    if not urls:
        logging.warning("No URLs provided to fetch data.")
        return []

    loader = UnstructuredURLLoader(urls=urls)
    try:
        docs = loader.load()  # Load the data from all URLs
        logging.info(f"Loaded {len(docs)} documents from URLs.")
        return docs
    except Exception as e:
        logging.error(f"Error loading documents from URLs: {e}")
        return []


# ✅ Split long documents into smaller overlapping text chunks
def split_text_chunks(documents: list, chunk_size=1000, chunk_overlap=200):
    if not documents:
        logging.warning("No documents to split.")
        return []

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} text chunks.")
    return chunks


# ✅ Load sentence embeddings using a HuggingFace model
def download_hugging_face_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logging.info(f"Loaded HuggingFace embeddings: {model_name}")
        return embeddings
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None

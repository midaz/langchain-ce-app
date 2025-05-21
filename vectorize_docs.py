import os
import tempfile
import ssl
import certifi
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
import math


#read local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 



# Configure requests to use certifi's certificates
requests.packages.urllib3.util.ssl_.DEFAULT_CERTS = certifi.where()

def get_vector_db_retriever():
    """Get or create a vector store retriever for Langchain documentation."""
    persist_path = os.path.join(tempfile.gettempdir(), "langchain_docs.parquet")
    print(f"Vector store path: {persist_path}")
    print(f"Vector store exists: {os.path.exists(persist_path)}")
    
    embd = OpenAIEmbeddings()

    # If vector store exists, then load it
    if os.path.exists(persist_path):
        print("Loading existing vector store...")
        try:
            vectorstore = SKLearnVectorStore(
                embedding=embd,
                persist_path=persist_path,
                serializer="parquet"
            )
            print("Successfully loaded existing vector store")
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 1,  # Number of documents to retrieve
                    "score_threshold": 0.9  # Minimum similarity score
                }
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Will create new vector store instead")

    print("Creating new vector store...")
    # Otherwise, index Langchain documents and create new vector store
    python_docs_loader = SitemapLoader(
        web_path="https://python.langchain.com/sitemap.xml",
        continue_on_failure=True,
        verify_ssl=False  # Disable SSL verification
    )
    
    print("Downloading Python documentation...")
    python_docs = python_docs_loader.load()
    print(f"Downloaded {len(python_docs)} Python documents")

    # Split documents using tiktoken encoder with better chunking strategy
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,  # Keep small chunk size to stay under token limits
        chunk_overlap=50,  # Small overlap for context while staying under limits
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # More natural text splitting
    )
    doc_splits = text_splitter.split_documents(python_docs)
    print(f"Created {len(doc_splits)} chunks")

    # Create and persist vector store
    print("Creating vector store...")
    
    # Initialize empty vector store
    vectorstore = SKLearnVectorStore(
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    
    # Process in smaller batches to handle token limits
    batch_size = 50  # Reduced batch size to stay well under token limits
    num_batches = math.ceil(len(doc_splits) / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(doc_splits))
        batch = doc_splits[start_idx:end_idx]
        print(f"Processing batch {i+1}/{num_batches} ({len(batch)} chunks)")
        
        try:
            # Add batch to vector store
            vectorstore.add_documents(batch)
            vectorstore.persist()
            print(f"Batch {i+1} processed and persisted")
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            print("Trying with smaller batch...")
            # If batch fails, try processing one document at a time
            for doc in batch:
                try:
                    vectorstore.add_documents([doc])
                    vectorstore.persist()
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
    
    print(f"Vector store saved to: {persist_path}")
    print(f"Vector store file exists: {os.path.exists(persist_path)}")
    print(f"Vector store file size: {os.path.getsize(persist_path) if os.path.exists(persist_path) else 'N/A'} bytes")
    
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,  # Number of documents to retrieve
            "score_threshold": 0.9  # Minimum similarity score
        }
    )

def main():
    # Get the retriever
    retriever = get_vector_db_retriever()
    
    # Test the retriever
    query = "How do I create a router chain in Langchain?"
    print(f"\nTesting retriever with query: {query}")
    
    # Use invoke instead of get_relevant_documents
    docs = retriever.invoke(query)
    print(f"Found {len(docs)} relevant documents")
    
    # Show better previews of the documents
    print("\nRelevant document previews:")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print("-" * 80)
        print(doc.page_content[:1000]) 
        print("-" * 80)
        if doc.metadata:
            print("Metadata:", doc.metadata)
        print()

if __name__ == "__main__":
    main() 
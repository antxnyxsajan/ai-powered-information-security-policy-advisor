import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. Configuration and Setup ---

# Load environment variables from a .env file located one directory up
#load_dotenv(dotenv_path='../.env')

# Pinecone and Namespace Configuration


PINECONE_INDEX_NAME = "security-advisor-index"
STANDARDS_NAMESPACE = "iso-nist-standards"


# --- 2. Core Functions ---

def load_and_split_txt(file_path: str):
    """
    Loads a TXT file and splits it into smaller, manageable documents.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return []

    print(f"Loading and splitting TXT from '{file_path}'...")
    # Use TextLoader for .txt files
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # Check if the document is large enough to need splitting
    # This part is optional but good practice
    first_doc_content = documents[0].page_content if documents else ""
    if len(first_doc_content) < 1000:
         print("Document is short, treating as a single chunk.")
         return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=100   # Characters to overlap between chunks for context
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} documents (chunks).")
    return docs


def index_documents(docs_to_index, namespace):
    """
    Embeds documents using Cohere and indexes them in a specified Pinecone namespace.
    """
    if not docs_to_index:
        print(f"No documents to index. Skipping.")
        return

    print(f"Indexing {len(docs_to_index)} documents into namespace '{namespace}'...")
    try:
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        PineconeVectorStore.from_documents(
            documents=docs_to_index,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace
        )
        print(f"Successfully indexed documents into '{namespace}'.")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")


# --- 3. Main Execution Block ---

if __name__ == '__main__':
    # Dynamically construct the path to the documents directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    docs_dir = os.path.join(project_root, 'documents')

    if not os.path.isdir(docs_dir):
        print(f"Error: Documents directory not found at '{docs_dir}'")
    else:
        for filename in os.listdir(docs_dir):
            if filename.endswith(".txt"):
                full_path = os.path.join(docs_dir, filename)
                
                print(f"\n--- Processing file: {filename} ---")
                
                documents = load_and_split_txt(full_path)

                # Create a new list of documents with cleaned metadata.
                cleaned_docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={'source': os.path.basename(doc.metadata['source'])}
                    ) for doc in documents
                ]
                
                # Index the documents with the cleaned metadata
                index_documents(cleaned_docs, STANDARDS_NAMESPACE)
            else:
                print(f"\n--- Skipping non-TXT file: {filename} ---")

    print("\n--- All processing complete. ---")
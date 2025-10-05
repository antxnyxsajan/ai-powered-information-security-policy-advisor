import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_INDEX_NAME = "security-advisor-index"
STANDARDS_NAMESPACE = "iso-nist-standards" 


def load_and_split_txt(file_path: str):
    """Loads a TXT file and splits it into smaller, manageable documents."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return []

    print(f"Loading and splitting TXT from '{file_path}'...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # Optional: If a document is short, don't split it
    if len(documents[0].page_content) < 1000:
        print("Document is short, treating as a single chunk.")
        return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} documents (chunks).")
    return docs


def index_documents(docs_to_index, namespace):
    """Embeds and indexes documents into a specified Pinecone namespace."""
    if not docs_to_index:
        print("No documents to index. Skipping.")
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


if __name__ == '__main__':
    # Define the directory containing ONLY standard documents
    # Example: ../documents/standards/
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    # This path now points specifically to a folder for standards
    standards_docs_dir = os.path.join(project_root, 'documents', 'standards')

    if not os.path.isdir(standards_docs_dir):
        print(f"Error: Standards directory not found at '{standards_docs_dir}'")
    else:
        print(f"--- Processing STANDARD documents from '{standards_docs_dir}' ---")
        for filename in os.listdir(standards_docs_dir):
            if filename.endswith(".txt"):
                full_path = os.path.join(standards_docs_dir, filename)
                print(f"\n--- Processing file: {filename} ---")
                
                documents = load_and_split_txt(full_path)

                # Clean metadata to only include the filename
                cleaned_docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={'source': os.path.basename(doc.metadata['source'])}
                    ) for doc in documents
                ]
                
                # Index the documents into the specific STANDARDS namespace
                index_documents(cleaned_docs, STANDARDS_NAMESPACE)
            else:
                print(f"\n--- Skipping non-TXT file: {filename} ---")

    print("\n--- Standard document processing complete. ---")
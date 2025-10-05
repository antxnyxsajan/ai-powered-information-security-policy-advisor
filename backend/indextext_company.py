import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Pinecone and Namespace Configuration
PINECONE_INDEX_NAME = "security-advisor-index"
COMPANY_DOCS_NAMESPACE = "company-internal-docs"


def load_and_split_txt(file_path: str):
    """Loads a TXT file and splits it into smaller, manageable documents."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return []

    print(f"Loading and splitting TXT from '{file_path}'...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
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
    # Define the directory containing ONLY company documents
    # Example: ../documents/company/
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    # This path now points specifically to a folder for company files
    company_docs_dir = os.path.join(project_root, 'documents', 'company')

    if not os.path.isdir(company_docs_dir):
        print(f"Error: Company documents directory not found at '{company_docs_dir}'")
    else:
        print(f"--- Processing COMPANY documents from '{company_docs_dir}' ---")
        for filename in os.listdir(company_docs_dir):
            if filename.endswith(".txt"):
                full_path = os.path.join(company_docs_dir, filename)
                print(f"\n--- Processing file: {filename} ---")
                
                documents = load_and_split_txt(full_path)

                cleaned_docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={'source': os.path.basename(doc.metadata['source'])}
                    ) for doc in documents
                ]
                
                # Index the documents into the specific COMPANY namespace
                index_documents(cleaned_docs, COMPANY_DOCS_NAMESPACE)
            else:
                print(f"\n--- Skipping non-TXT file: {filename} ---")

    print("\n--- Company document processing complete. ---")
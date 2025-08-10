from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv(dotenv_path='../key.env')

PINECONE_INDEX_NAME = "security-advisor-index"
STANDARDS_NAMESPACE = "iso-nist-standards"
COMPANY_POLICY_NAMESPACE = "company-xyz-policy"

# Sample Company Policies

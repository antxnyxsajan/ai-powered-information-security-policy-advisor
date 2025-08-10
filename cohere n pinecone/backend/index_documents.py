import os 

load_dotenv(dotenv_path='../.env')

#for testing 
#os.environ["COHERE_API_KEY"] = "your_cohere_api_key_goes_here"
#os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key_goes_here"

PINECONE_INDEX_NAME = "security-advisor-index"
STANDARDS_NAMESPACE = "iso-nist-standards"
COMPANY_POLICY_NAMESPACE = "company-xyz-policy"

# Sample Company Policies
company_policy_texts = [
    "The Work From Home (WFH) policy allows employees to work remotely two days a week, on Monday and Friday.",
    "All employees must use a password manager and enable two-factor authentication (2FA) on all company accounts.",
    "Use of personal devices to access company data is permitted, but the device must be encrypted and have anti-virus software installed."
]

# Sample General Standards
standards_texts = [
    "NIST guidelines recommend that passwords be at least 12 characters long and include a mix of uppercase, lowercase, numbers, and symbols.",
    "ISO 27001 is an international standard for information security management.",
    "Regular security awareness training for all employees is a key component of a robust security posture."
]

company_documents = [Document(page_content=text) for text in company_policy_texts]
standards_documents = [Document(page_content=text) for text in standards_texts]

def index_documents(docs_to_index, namespace):
    print(f"Indexing {len(docs_to_index)} documents into namespace '{namespace}'...")
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    PineconeVectorStore.from_documents(
        documents=docs_to_index,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace
    )
    print("Done.")

if __name__ == '__main__':
    index_documents(
        docs_to_index=company_documents,
        namespace=COMPANY_POLICY_NAMESPACE
    )
    index_documents(
        docs_to_index=standards_documents,
        namespace=STANDARDS_NAMESPACE
    )
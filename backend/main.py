# backend/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_cohere import ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate

# Securely load API keys from your .env file
load_dotenv()

# --- Initialize Models and Retrievers ---
llm = ChatCohere(model="command-a-03-2025")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name="security-advisor-index", embedding=embeddings)

# --- THE FIX: Update Retriever to use a score threshold ---
# This only returns documents that have a similarity score above 0.5.
# This makes the fallback logic work correctly.
company_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'score_threshold': 0.5, 'namespace': 'company-internal-docs'}
)
standards_retriever = vectorstore.as_retriever(search_kwargs={'namespace': 'iso-nist-standards'})

prompt_template = """
You are an AI assistant for company employees. Your tone should be helpful, professional, and clear.
Answer the user's question based ONLY on the context provided below.

**CRITICAL INSTRUCTIONS FOR FORMATTING YOUR RESPONSE:**
- Use **Markdown** for all formatting.
- Use headings (e.g., `### Section Title`) for different sections of your answer.
- If the answer is a detailed explanation of a policy, start with a relevant heading (e.g., `### Work From Home Policy`).
- For simple questions or greetings, provide a direct, conversational answer without a heading.
- Use bullet points (`-`) for lists or steps. You can use nested bullets for sub-points.
- Use bold text (`**key term**`) to highlight important concepts.
- Write in clear, well-structured paragraphs.
- If the context does not contain the answer, you MUST say: "I could not find information on that topic in the provided documents."
- Dont over crowd responses keep it short if possible but without risking necessary information.
- Interpret if the queryt is a question or a simple chat and answer appropriately dont include rule 5 if the query is a chat prompt and not necessarily a question regarding policy or company.


Context:
{context}

Question:
{question}
"""

company_prompt = ChatPromptTemplate.from_template(prompt_template)
standards_prompt = ChatPromptTemplate.from_template(prompt_template)

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# In backend/main.py, replace the entire @app.post("/chat") function

# In backend/main.py, replace the entire @app.post("/chat") function

@app.post("/chat")
def chat_handler(request: ChatRequest):
    print(f"Received question: {request.question}")

    lowered_question = request.question.lower().strip()
    greetings = ["hello", "hi", "hey", "yo", "hai"]
    if lowered_question in greetings:
        print("Handling as a simple greeting.")
        return {"answer": "Hello! How can I help you with your policy questions today?", "source": ""}
    
    print("\n--- Checking Company Policy Docs ---")
    
    company_docs_with_scores = vectorstore.similarity_search_with_score(
        request.question, 
        namespace='company-internal-docs',
        k=5
    )
    
    for doc, score in company_docs_with_scores:
        print(f"Initial Company Doc Score: {score:.4f}")

    score_threshold = 0.55
    company_docs = []
    
    high_score_docs = [doc for doc, score in company_docs_with_scores if score >= score_threshold]

    if high_score_docs:
        print(f"\nFound {len(high_score_docs)} docs above score threshold.")
        company_docs = high_score_docs
    else:
        print(f"\nNo docs found above threshold. Checking for keywords in lower-scoring docs.")
        priority_keywords = ["company", "policy", "standard", "general", "internal"]
        keyword_boosted_docs = []
        for doc, score in company_docs_with_scores:
            doc_text_lower = doc.page_content.lower()
            if any(keyword in doc_text_lower for keyword in priority_keywords):
                print(f"Found keyword in doc with score {score:.4f}. Boosting it.")
                keyword_boosted_docs.append(doc)
        
        if keyword_boosted_docs:
            company_docs = keyword_boosted_docs

    if company_docs:
        print(f"\nFound {len(company_docs)} relevant docs in Company Policy. Using company chain.")
        context = "\n\n".join([doc.page_content for doc in company_docs])
        chain = company_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})
        return {"answer": response.content, "source": "Company Policy"}
    else:
        print(f"\nNo relevant company docs found. Falling back to standards.")
        
        print("\n--- Checking Standards Docs ---")
        standards_docs_with_scores = vectorstore.similarity_search_with_score(
            request.question,
            namespace='iso-nist-standards',
            k=4
        )
        
        if not standards_docs_with_scores:
            print("No documents found in Standards at all.")
            return {"answer": "I could not find any relevant information in the company policy or the general standards.", "source": ""}

        for doc, score in standards_docs_with_scores:
            print(f"Standard Doc Score: {score:.4f}")
            
        source = "General Standards"
        fallback_threshold = 0.45
        
        best_fallback_score = standards_docs_with_scores[0][1]
        if best_fallback_score < fallback_threshold:
            print(f"Best fallback score ({best_fallback_score:.4f}) is below threshold {fallback_threshold}. Clearing source.")
            source = ""
            
        standards_docs = [doc for doc, score in standards_docs_with_scores]
        
        print(f"Found {len(standards_docs)} docs in Standards. Using standards chain.")
        context = "\n\n".join([doc.page_content for doc in standards_docs])
        chain = standards_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})
        
        return {"answer": response.content, "source": source}
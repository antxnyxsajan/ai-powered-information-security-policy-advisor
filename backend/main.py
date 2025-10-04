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
#load_dotenv()
check env file next time i think its working
# --- Initialize Models and Retrievers ---
llm = ChatCohere(model="command-a-03-2025")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name="security-advisor-index", embedding=embeddings)

# --- THE FIX: Update Retriever to use a score threshold ---
# This only returns documents that have a similarity score above 0.5.
# This makes the fallback logic work correctly.
company_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'score_threshold': 0.5, 'namespace': 'company-xyz-policy'}
)
standards_retriever = vectorstore.as_retriever(search_kwargs={'namespace': 'iso-nist-standards'})

# --- Improved Prompt for Structured Responses ---
prompt_template = """
You are an AI assistant for company employees. Your tone should be helpful and professional.
Answer the user's question based ONLY on the context provided below.

Follow these rules for your response:
1.  Use simple, clear language that is easy to understand.
2.  If the answer involves a list, steps, or multiple points, use bullet points.
3.  Highlight key terms or action items by making them **bold**.
4.  Keep the answer concise and directly related to the question.
5.  If the context does not contain the answer, you MUST say: "I could not find information on that topic in the provided documents."
6.  Dont over crowd responses keep it short if possible but without risking necessary information.
7.  Interpret if the queryt is a question or a simple chat and answer appropriately dont include rule 5 if the query is a chat prompt and not necessarily a question regarding policy or company.

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

@app.post("/chat")
def chat_handler(request: ChatRequest):
    print(f"Received question: {request.question}")

    # --- NEW: Handle small talk ---
    lowered_question = request.question.lower()
    greetings = ["hello", "hi", "hey", "yo"]
    if any(greeting in lowered_question for greeting in greetings) and len(lowered_question.split()) < 3:
        return {"answer": "Hello! How can I help you with your policy questions today?"}
    
    # --- RAG Logic ---
    # The retriever now only returns documents if they are relevant enough
    company_docs = company_retriever.invoke(request.question)
    
    if company_docs:
        print("Found relevant company policy. Using company chain.")
        context = "\n\n".join([doc.page_content for doc in company_docs])
        chain = company_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})
    else:
        print("No company policy found. Falling back to standards.")
        standards_docs = standards_retriever.invoke(request.question)
        if not standards_docs:
            return {"answer": "I could not find any relevant information in the company policy or the general standards. Please ask another question."}
        context = "\n\n".join([doc.page_content for doc in standards_docs])
        chain = standards_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})

    return {"answer": response.content}
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

#load_dotenv(dotenv_path='../.env')

#for testing

llm = ChatCohere(model="command-r")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name="security-advisor-index", embedding=embeddings)

company_retriever = vectorstore.as_retriever(search_kwargs={'namespace': 'company-xyz-policy'})
standards_retriever = vectorstore.as_retriever(search_kwargs={'namespace': 'iso-nist-standards'})

#Prompts
company_prompt = ChatPromptTemplate.from_template(
    "Answer based on this specific Company Policy:\n\n{context}\n\nQuestion: {question}"
)
standards_prompt = ChatPromptTemplate.from_template(
    "A specific company policy was not found. Answer based on these general standards:\n\n{context}\n\nQuestion: {question}"
)

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
    #This is our conditional retrieval logic
    print(f"Received question: {request.question}")
    company_docs = company_retriever.get_relevant_documents(request.question)
    
    if company_docs:
        print("Found relevant company policy. Using company chain.")
        context = "\n\n".join([doc.page_content for doc in company_docs])
        chain = company_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})
    else:
        print("No company policy found. Falling back to standards.")
        standards_docs = standards_retriever.get_relevant_documents(request.question)
        if not standards_docs:
            return {"answer": "I could not find any relevant information in the company policy or the general standards. Please ask another question."}
        context = "\n\n".join([doc.page_content for doc in standards_docs])
        chain = standards_prompt | llm
        response = chain.invoke({"context": context, "question": request.question})

    return {"answer": response.content}
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_cohere import ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatCohere(model="command-a-03-2025")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name="security-advisor-index", embedding=embeddings)

company_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'score_threshold': 0.55, 'namespace': 'company-internal-docs'}
)
standards_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.45, 'namespace': 'iso-nist-standards'}
)

# --- PROMPT 1: For detailed RAG questions ---
rag_prompt_template = """
You are an AI assistant for company employees. Your tone should be helpful, professional, and clear.
Answer the user's question based ONLY on the context provided below.

**CRITICAL INSTRUCTIONS FOR FORMATTING YOUR RESPONSE:**
- If the context is empty, you MUST politely state that you couldn't find a specific policy related to the user's question and ask them to rephrase or ask about another topic. Do not mention the word "context".
- If the answer is a detailed explanation of a policy, start with a relevant heading (e.g., `### Work From Home Policy`).
- For simple questions or greetings, provide a direct, conversational answer without a heading.
- Use bullet points (`-`) for lists or steps.
- Use bold text (`**key term**`) to highlight important concepts.
- Write in clear, well-structured paragraphs.

Context:
{context}

Question:
{question}
"""

# --- PROMPT 2: For simple conversational chat ---
chat_prompt_template = """
You are a helpful and professional AI Security Advisor.
A user has sent you a conversational message. Respond in a brief, friendly, and professional manner.
If they are thanking you, acknowledge it warmly. If they are greeting you, greet them back and ask how you can help with their policy questions.

User's message:
{question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
chat_prompt = ChatPromptTemplate.from_template(chat_prompt_template)


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

    lowered_question = request.question.lower().strip()
    conversational_phrases = [
        "hello", "hi", "hey", "yo", "hai",
        "how are you", "how are you doing", "what's up",
        "thank you", "thanks", "thx",
        "good work", "great job", "awesome", "perfect", "ok", "cool"
    ]

    if lowered_question in conversational_phrases:
        print("Handling as a conversational query. Using chat prompt.")
        chat_chain = chat_prompt | llm
        response = chat_chain.invoke({"question": request.question})
        return {"answer": response.content, "source": ""}
    
    print("Handling as a policy question. Using RAG chain.")
    context = ""
    source = ""
    
    print("\n--- Checking Company Policy Docs ---")
    company_docs_with_scores = vectorstore.similarity_search_with_score(
        request.question, 
        namespace='company-internal-docs',
        k=4
    )
    
    for doc, score in company_docs_with_scores:
        print(f"Company Doc Score: {score:.4f}")

    score_threshold = 0.45
    company_docs = [doc for doc, score in company_docs_with_scores if score >= score_threshold]

    if company_docs:
        print(f"\nFound {len(company_docs)} relevant docs in Company Policy.")
        context = "\n\n".join([doc.page_content for doc in company_docs])
        source = "Company Policy"
    else:
        print(f"\nNo relevant company docs found. Falling back to standards.")
        
        standards_docs_with_scores = vectorstore.similarity_search_with_score(
            request.question,
            namespace='iso-nist-standards',
            k=4
        )
        
        for doc, score in standards_docs_with_scores:
            print(f"Standard Doc Score: {score:.4f}")
        
        fallback_threshold = 0.4
        standards_docs = [doc for doc, score in standards_docs_with_scores if score >= fallback_threshold]
        
        if standards_docs:
            print(f"Found {len(standards_docs)} relevant docs in Standards.")
            context = "\n\n".join([doc.page_content for doc in standards_docs])
            source = "General Standards"
        else:
            print("No relevant docs found in Standards either.")

    print(f"Final context length: {len(context)}, Source: '{source}'")
    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"context": context, "question": request.question})
    
    if "I could not find information" in response.content:
        source = ""
    
    return {"answer": response.content, "source": source}
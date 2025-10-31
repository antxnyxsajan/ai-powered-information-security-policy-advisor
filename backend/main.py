import os
import time
import psutil  # <-- NEW: Import for memory usage
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
from typing import List, Dict, Tuple, Any

from langchain_cohere import ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# --- NEW: Get process for memory tracking ---
PROCESS = psutil.Process(os.getpid())
# --- End new ---

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

# --- Helper function to calculate averages ---
def _calculate_avg(times_list: List[float]) -> Tuple[float, int]:
    """Calculates the average and count from a list of times."""
    count = len(times_list)
    if count == 0:
        return 0.0, 0
    avg = sum(times_list) / count
    return avg, count
# --- End ---


# --- MODIFIED: App startup event to initialize all metrics ---
@app.on_event("startup")
async def startup_event():
    """On server startup, create a dictionary of lists to store all metric times."""
    app.state.metrics: Dict[str, Any] = {
        # Performance metrics
        "total": [],
        "vdb1": [],
        "vdb2": [],
        "llm": [],
        # RAG Quality metrics
        "scores": [],
        "doc_counts": [],
        "context_lengths": [],
        "hits_company": 0,
        "hits_standards": 0,
        "hits_none": 0
    }
# --- End modified ---


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODIFIED: Performance Logging Middleware ---
@app.middleware("http")
async def log_performance_metrics(request: Request, call_next):
    """
    This middleware now handles all performance logging to avoid duplicates
    and calculates the runtime average for ALL tracked metrics.
    """
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate total time
    process_time = time.time() - start_time
    
    # Only log stats for the /chat endpoint
    if request.url.path == "/chat":
        # === PERFORMANCE ===
        app.state.metrics["total"].append(process_time)
        
        print("\n--- PERFORMANCE ANALYSIS ---")
        
        # --- VDB 1 ---
        if hasattr(request.state, "vdb_time_1"):
            app.state.metrics["vdb1"].append(request.state.vdb_time_1)
            vdb1_avg, vdb1_count = _calculate_avg(app.state.metrics["vdb1"])
            print(f"[VDB 1 (Company)] Current: {request.state.vdb_time_1:<7.4f}s | Average ({vdb1_count} reqs): {vdb1_avg:.4f}s")
        
        # --- VDB 2 (Fallback) ---
        if hasattr(request.state, "vdb_time_2"):
            app.state.metrics["vdb2"].append(request.state.vdb_time_2)
        
        vdb2_avg, vdb2_count = _calculate_avg(app.state.metrics["vdb2"])
        if vdb2_count > 0:
            current_vdb2 = f"{request.state.vdb_time_2:<7.4f}s" if hasattr(request.state, "vdb_time_2") else " " * 7
            print(f"[VDB 2 (Standards)] Current: {current_vdb2} | Average ({vdb2_count} reqs): {vdb2_avg:.4f}s")

        # --- LLM Generation ---
        if hasattr(request.state, "llm_time"):
            app.state.metrics["llm"].append(request.state.llm_time)
            llm_avg, llm_count = _calculate_avg(app.state.metrics["llm"])
            print(f"[LLM Generation]  Current: {request.state.llm_time:<7.4f}s | Average ({llm_count} reqs): {llm_avg:.4f}s")
        
        print("---------------------------------")
        # --- Total Runtime ---
        total_avg, total_count = _calculate_avg(app.state.metrics["total"])
        print(f"[Total Runtime]     Current: {process_time:<7.4f}s | Average ({total_count} reqs): {total_avg:.4f}s")

        # === RAG QUALITY (Only if it was a RAG query) ===
        if hasattr(request.state, "hit_type"):
            # --- Handle Hit Counters ---
            if request.state.hit_type == "company":
                app.state.metrics["hits_company"] += 1
            elif request.state.hit_type == "standards":
                app.state.metrics["hits_standards"] += 1
            elif request.state.hit_type == "none":
                app.state.metrics["hits_none"] += 1
            
            # --- Append other RAG stats ---
            app.state.metrics["scores"].append(request.state.retrieval_score)
            app.state.metrics["doc_counts"].append(request.state.doc_count)
            app.state.metrics["context_lengths"].append(request.state.context_length)

            # --- Calculate Averages ---
            score_avg, score_count = _calculate_avg(app.state.metrics["scores"])
            docs_avg, docs_count = _calculate_avg(app.state.metrics["doc_counts"])
            len_avg, len_count = _calculate_avg(app.state.metrics["context_lengths"])
            
            # --- Get Current Values ---
            current_score = request.state.retrieval_score
            current_docs = request.state.doc_count
            current_len = request.state.context_length
            current_mem_mb = PROCESS.memory_info().rss / (1024 * 1024)
            
            # --- Print RAG Dashboard ---
            print("\n--- RAG QUALITY & RESOURCES ---")
            hits_c = app.state.metrics["hits_company"]
            hits_s = app.state.metrics["hits_standards"]
            hits_n = app.state.metrics["hits_none"]
            print(f"[Retrieval Hits]  Company: {hits_c} | Standards: {hits_s} | None: {hits_n}")
            print(f"[Retrieval Score] Current: {current_score:<7.4f}  | Average ({score_count} reqs): {score_avg:.4f}")
            print(f"[Docs Retrieved]  Current: {current_docs:<7} | Average ({docs_count} reqs): {docs_avg:.2f}")
            print(f"[Context Length]  Current: {current_len:<7} | Average ({len_count} reqs): {len_avg:.0f} chars")
            print(f"[Memory Usage]    Current: {current_mem_mb:.2f} MB")
        
        print("---------------------------------\n")
    
    return response
# --- End of modified middleware ---


class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_handler(chat_request: ChatRequest, request: Request):
    print(f"Received question: {chat_request.question}")

    lowered_question = chat_request.question.lower().strip()
    conversational_phrases = [
        "hello", "hi", "hey", "yo", "hai",
        "how are you", "how are you doing", "what's up",
        "thank you", "thanks", "thx",
        "good work", "great job", "awesome", "perfect", "ok", "cool"
    ]

    if lowered_question in conversational_phrases:
        print("Handling as a conversational query. Using chat prompt.")
        
        llm_start = time.time()
        chat_chain = chat_prompt | llm
        response = chat_chain.invoke({"question": chat_request.question})
        request.state.llm_time = time.time() - llm_start
        
        return {"answer": response.content, "source": ""}
    
    print("Handling as a policy question. Using RAG chain.")
    
    context = ""
    source = ""
    
    print("\n--- Checking Company Policy Docs ---")
    
    vdb_start_1 = time.time()
    company_docs_with_scores = vectorstore.similarity_search_with_score(
        chat_request.question,
        namespace='company-internal-docs',
        k=4
    )
    vdb_time_1 = time.time() - vdb_start_1
    request.state.vdb_time_1 = vdb_time_1
    
    for doc, score in company_docs_with_scores:
        print(f"Company Doc Score: {score:.4f}")

    score_threshold = 0.45
    company_docs = [doc for doc, score in company_docs_with_scores if score >= score_threshold]

    if company_docs:
        print(f"\nFound {len(company_docs)} relevant docs in Company Policy.")
        context = "\n\n".join([doc.page_content for doc in company_docs])
        source = "Company Policy"
        
        # --- NEW: Set RAG stats for middleware ---
        request.state.hit_type = "company"
        request.state.retrieval_score = company_docs_with_scores[0][1] # Get top score
        request.state.doc_count = len(company_docs)
        # --- End new ---
        
    else:
        print(f"\nNo relevant company docs found. Falling back to standards.")
        
        vdb_start_2 = time.time()
        standards_docs_with_scores = vectorstore.similarity_search_with_score(
            chat_request.question,
            namespace='iso-nist-standards',
            k=4
        )
        vdb_time_2 = time.time() - vdb_start_2
        request.state.vdb_time_2 = vdb_time_2
        
        for doc, score in standards_docs_with_scores:
            print(f"Standard Doc Score: {score:.4f}")
        
        fallback_threshold = 0.4
        standards_docs = [doc for doc, score in standards_docs_with_scores if score >= fallback_threshold]
        
        if standards_docs:
            print(f"Found {len(standards_docs)} relevant docs in Standards.")
            context = "\n\n".join([doc.page_content for doc in standards_docs])
            source = "General Standards"

            # --- NEW: Set RAG stats for middleware ---
            request.state.hit_type = "standards"
            request.state.retrieval_score = standards_docs_with_scores[0][1] # Get top score
            request.state.doc_count = len(standards_docs)
            # --- End new ---

        else:
            print("No relevant docs found in Standards either.")
            
            # --- NEW: Set RAG stats for middleware ---
            request.state.hit_type = "none"
            request.state.retrieval_score = 0.0
            request.state.doc_count = 0
            # --- End new ---

    # --- NEW: Set context length for middleware ---
    request.state.context_length = len(context)
    # --- End new ---

    print(f"Final context length: {len(context)}, Source: '{source}'")
    
    llm_start = time.time()
    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"context": context, "question": chat_request.question})
    llm_time = time.time() - llm_start
    request.state.llm_time = llm_time
    
    if "I could not find information" in response.content:
        source = ""
    
    return {"answer": response.content, "source": source}
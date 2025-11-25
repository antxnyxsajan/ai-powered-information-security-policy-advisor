# backend/measure_metrics.py
import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
load_dotenv()
llm = ChatCohere(model="command-a-03-2025", temperature=0)
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name="security-advisor-index", embedding=embeddings)

# 2. Define The Test Set (Ground Truth)
# REPLACE these with REAL facts from your uploaded PDFs!
test_set = [
    {
        "question": "What is the maximum size limit for email attachments?",
        "required_keyword": "20 mb",  # Fact from Source [836]
        "negative_keyword": "25 mb"   # Common default (Gmail/Outlook) that generic AI might guess
    },
    {
        "question": "How many days is the password expiry ageing limit set to?",
        "required_keyword": "120 days", # Fact from Source [1676]
        "negative_keyword": "90 days"   # Standard industry practice generic AI might guess
    },
    {
        "question": "What is the retention period for CCTV surveillance footage?",
        "required_keyword": "15 days",  # Fact from Source [1302]
        "negative_keyword": "30 days"   # Common standard generic AI might guess
    },
    {
        "question": "After how many minutes of inactivity will the screen lock automatically?",
        "required_keyword": "5 minutes", # Fact from Source [932]
        "negative_keyword": "10 minutes" # Common default generic AI might guess
    },
    {
        "question": "Within what timeframe must critical security patches be applied?",
        "required_keyword": "1 week",    # Fact from Source [1112]
        "negative_keyword": "24 hours"   # Generic "urgent" guess
    },
    {
        "question": "What is the account lockout threshold for invalid login attempts?",
        "required_keyword": "5 attempts", # Fact from Source [1673]
        "negative_keyword": "3 attempts"  # Common strict standard generic AI might guess
    },
    {
        "question": "What is the reporting timeframe for Cert-In security incidents?",
        "required_keyword": "6 hours",   # Fact from Source [863]
        "negative_keyword": "24 hours"   # Standard GDPR/Industry timeframe a generic AI guesses
    },
    {
        "question": "What is the maximum time allowed for suppliers to report security incidents?",
        "required_keyword": "8 hrs",     # Fact from Source [1349]
        "negative_keyword": "immediately" # Generic vague answer
    },
    {
        "question": "How often must the leadership team conduct management review meetings?",
        "required_keyword": "quarterly", # Fact from Source [1249]
        "negative_keyword": "annually"   # ISO 27001 standard minimum is often interpreted as annual
    },
    {
        "question": "Which specific internet browsers are authorized for use?",
        "required_keyword": "microsoft edge", # Fact from Source [644] (Mentions Edge & Chrome)
        "negative_keyword": "firefox"         # A standard AI would assume Firefox is also allowed
    },
    {
        "question": "How many pillars of Cyber Resiliency are defined in the incident policy?",
        "required_keyword": "six",       # Fact from Source [845] (RSET uses 6: Predict, Prepare, Protect, Detect, Respond, Recover)
        "negative_keyword": "five"       # NIST Framework standard has 5 (Ident, Prot, Det, Resp, Rec)
    },
    {
        "question": "What are the three defined tiers for information asset classification?",
        "required_keyword": "internal",  # Fact from Source [769] (Confidential, Internal, Public)
        "negative_keyword": "restricted" # Common industry tier that RSET does NOT use
    },
    {
        "question": "How many building blocks are defined in the Information Security strategy?",
        "required_keyword": "4 building blocks", 
        "negative_keyword": "3 building blocks" # Generic guess
    },
    {
        "question": "What is the guiding principle for information security awareness?",
        "required_keyword": "incomplete without you", # Phrase: "Security is... incomplete without you"
        "negative_keyword": "watchfulness"
    },
    {
        "question": "Who is responsible for appointing the CISO?",
        "required_keyword": "ceo", 
        "negative_keyword": "board of directors"
    },
    {
        "question": "Who is responsible for the classification of information assets?",
        "required_keyword": "information asset owner", 
        "negative_keyword": "it manager"
    },
    {
        "question": "Which specific locations are covered in the ISMS Scope?",
        "required_keyword": "trivandrum and kochi", 
        "negative_keyword": "all locations"
    },

    # --- TRAINING & AWARENESS ---
    {
        "question": "How soon must new staff start security awareness training?",
        "required_keyword": "within a week", 
        "negative_keyword": "within 30 days"
    },
    {
        "question": "What is the frequency of refresher training for all staff?",
        "required_keyword": "once in a year", 
        "negative_keyword": "quarterly"
    },

    # --- DATA & OPERATIONS ---
    {
        "question": "Which specific cloud service is mandated for user data backup?",
        "required_keyword": "onedrive", 
        "negative_keyword": "google drive"
    },
    {
        "question": "What must be done to whiteboards in meeting rooms after use?",
        "required_keyword": "cleaned", 
        "negative_keyword": "erased"
    },
    {
        "question": "Are visitors allowed to connect to the internal network?",
        "required_keyword": "not be allowed", 
        "negative_keyword": "guest network"
    },
    {
        "question": "What is the default setting for access control systems?",
        "required_keyword": "deny-all", 
        "negative_keyword": "allow-all"
    },
    {
        "question": "How often must user access in systems be reviewed?",
        "required_keyword": "once in a year", 
        "negative_keyword": "every 6 months"
    },

    # --- PATCHING & INFRASTRUCTURE ---
    {
        "question": "What is the SLA for applying 'Important' security patches?",
        "required_keyword": "4 weeks", 
        "negative_keyword": "1 week"
    },
    {
        "question": "How often is it mandatory to restart endpoint devices?",
        "required_keyword": "once a week", 
        "negative_keyword": "daily"
    },
    {
        "question": "To prevent flood risk, where should the office be located?",
        "required_keyword": "upper floors", 
        "negative_keyword": "ground floor"
    },
    {
        "question": "Who must authorize taking information assets off premises?",
        "required_keyword": "admin head", 
        "negative_keyword": "security guard"
    },
    {
        "question": "Are smart phones allowed inside the clean room?",
        "required_keyword": "not be allowed", 
        "negative_keyword": "silent mode"
    },
    {
        "question": "How often must firewall configurations be reviewed?",
        "required_keyword": "annually", 
        "negative_keyword": "monthly"
    },
    {
        "question": "Can remote employees use public Wi-Fi networks?",
        "required_keyword": "must not use", 
        "negative_keyword": "with vpn"
    },

    # --- THIRD PARTY & COMPLIANCE ---
    {
        "question": "What specific agreement must be signed by a supplier before engagement?",
        "required_keyword": "non-disclosure & service agreement", 
        "negative_keyword": "nda"
    },
    {
        "question": "Does PII include the business telephone number of an employee?",
        "required_keyword": "does not include", 
        "negative_keyword": "yes"
    },
    {
        "question": "Name one specific agency listed for threat intelligence sources.",
        "required_keyword": "nasscom", 
        "negative_keyword": "fbi"
    },

    # --- PASSWORDS & SECURITY TESTING ---
    {
        "question": "What is the minimum character length for passwords?",
        "required_keyword": "8 characters", 
        "negative_keyword": "10 characters"
    },
    {
        "question": "How often must penetration testing be conducted?",
        "required_keyword": "once in a year", 
        "negative_keyword": "twice a year"
    },
    {
        "question": "What specific plan must be created before any patch rollout?",
        "required_keyword": "rollback plan", 
        "negative_keyword": "backup plan"
    },

    # --- DOCUMENT CONTROL FACTS ---
    {
        "question": "Who must approve any distribution of private documents outside the company?",
        "required_keyword": "ciso", 
        "negative_keyword": "manager"
    },
    {
        "question": "What is the specific channel for publishing the ISMS document?",
        "required_keyword": "teams@rset", 
        "negative_keyword": "email"
    },
    {
        "question": "What is the current version number of the ISMS Policy?",
        "required_keyword": "version 2.2", 
        "negative_keyword": "1.0"
    },
    {
        "question": "On what date was the 'Computing and Mobile devices usage' amended (Version 1.1)?",
        "required_keyword": "12-jan-2020", 
        "negative_keyword": "2021"
    },
    {
        "question": "Which two browsers are authorized for use?",
        "required_keyword": "microsoft edge", 
        "negative_keyword": "safari" 
    }
]

# 3. Define the Contenders
# --- Standard Bot (No Context) ---
std_prompt = ChatPromptTemplate.from_template("Answer the user's question: {question}")
std_chain = std_prompt | llm | StrOutputParser()

# --- Your RAG Bot ---
def get_rag_answer(q):
    docs = vectorstore.similarity_search(q, k=4, namespace='company-internal-docs')
    context = "\n".join([d.page_content for d in docs])
    
    rag_prompt = ChatPromptTemplate.from_template(
        "Answer based ONLY on this context:\n{context}\n\nQuestion: {question}"
    )
    chain = rag_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"context": context, "question": q})
    except:
        return "Error"

# 4. Run the Experiment
results = []

print(f"{'='*10} RUNNING MEASURABLE METRICS TEST {'='*10}\n")

for item in test_set:
    q = item["question"]
    truth = item["required_keyword"]
    
    print(f"Testing: {q}")
    
    # Get Answers
    ans_std = std_chain.invoke({"question": q}).lower()
    ans_rag = get_rag_answer(q).lower()
    
    # Measure: Standard Bot
    std_correct = 1 if truth in ans_std else 0
    std_hallucinated = 1 if item.get("negative_keyword") in ans_std else 0
    
    # Measure: RAG Bot
    rag_correct = 1 if truth in ans_rag else 0
    rag_hallucinated = 1 if item.get("negative_keyword") in ans_rag else 0
    
    results.append({
        "Question": q,
        "Std_Correct": std_correct,
        "Std_Hallucinated": std_hallucinated,
        "RAG_Correct": rag_correct,
        "RAG_Hallucinated": rag_hallucinated
    })

# 5. Calculate Final Metrics
df = pd.DataFrame(results)

print("\n" + "="*30)
print("FINAL MEASURABLE RESULTS")
print("="*30)
print(f"Total Questions: {len(test_set)}")
print(f"Standard Bot Accuracy: {df['Std_Correct'].mean():.0%}")
print(f"RAG Bot Accuracy:      {df['RAG_Correct'].mean():.0%}")
print(f"Standard Bot Hallucination Rate: {df['Std_Hallucinated'].mean():.0%}")
print(f"RAG Bot Hallucination Rate:      {df['RAG_Hallucinated'].mean():.0%}")
print("="*30)

# Save to CSV for your paper
df.to_csv("metric_results.csv", index=False)
print("\nDetailed results saved to 'metric_results.csv'")
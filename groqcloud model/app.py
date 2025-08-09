import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv("key.env")

llm=ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

st.title("InfoSec Policy Advisor")

query=st.text_input("Ask a question about security policy....")

if query:
    messages=[
        SystemMessage(content="You are an information security policy advisor trained on ISO 27001 and NIST standards."),
        HumanMessage(content=query)
    ]
    response=llm(messages)
    st.write(response.content)
import os
from dotenv import load_dotenv

env_path = '../key.env'
print(f"Attempting to load environment variables from: {os.path.abspath(env_path)}")

load_dotenv(dotenv_path=env_path)

cohere_key = os.getenv("COHERE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

print("\n--- DIAGNOSTIC RESULTS ---")

if cohere_key:
    print("✅ Cohere API Key: Found!")
else:
    print("❌ Cohere API Key: NOT FOUND.")

if pinecone_key:
    print("✅ Pinecone API Key: Found!")
else:
    print("❌ Pinecone API Key: NOT FOUND.")

print("--------------------------")
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

NAMESPACE_TO_UPDATE = "iso-nist-standards"
TEXT_TO_ADD = "\nSource: ISO 27001 external standards"
METADATA_TEXT_FIELD = "text" 

def update_single_vector(index, vector_id, namespace, text_to_add, metadata_field):
    try:
        fetch_response = index.fetch(ids=[vector_id], namespace=namespace)
        if not fetch_response.vectors:
            print(f"  !! WARNING: Could not fetch data for vector {vector_id}. Skipping.")
            return False
        
        vector_data = fetch_response.vectors[vector_id]

        original_metadata = vector_data.metadata
        original_text = original_metadata.get(metadata_field, "")
        updated_metadata = original_metadata.copy()
        updated_metadata[metadata_field] = original_text + text_to_add

        vector_to_upsert = {
            "id": vector_id,
            "values": vector_data.values,
            "metadata": updated_metadata
        }

        index.upsert(vectors=[vector_to_upsert], namespace=namespace)
        return True

    except Exception as e:
        print(f"  !! FAILED to process vector {vector_id}. Error: {e}")
        return False

def main():
    print("="*50)
    print("WARNING: This script will overwrite existing data in Pinecone.")
    print(f"It will modify all records in the '{NAMESPACE_TO_UPDATE}' namespace.")
    print("="*50)
    proceed = input("Type 'yes' to continue: ")

    if proceed.lower() != 'yes':
        print("Operation cancelled.")
        return

    print("Initializing Pinecone client...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("security-advisor-index")
    print(f"Connected to index 'security-advisor-index'.")

    # --- THE DEFINITIVE FIX: Use a nested loop to unpack the pages ---
    print(f"Fetching all vector IDs from namespace '{NAMESPACE_TO_UPDATE}'...")
    try:
        vector_ids = []
        # 'page_of_ids' will be a list like ['id-1', 'id-2', ...]
        for page_of_ids in index.list(namespace=NAMESPACE_TO_UPDATE, limit=100):
            # Then loop through each ID in that page and add it to our master list
            for single_id in page_of_ids:
                vector_ids.append(single_id)
    except Exception as e:
        print(f"An error occurred while listing vectors: {e}")
        return

    print(f"Successfully fetched {len(vector_ids)} vector IDs.") # This will now be 155
    if not vector_ids:
        print("No vectors found. Exiting.")
        return

    print("\nProcessing vectors one by one...")
    success_count = 0
    failure_count = 0
    for i, vector_id in enumerate(vector_ids):
        print(f"Processing vector {i + 1} of {len(vector_ids)} (ID: {vector_id})...")
        success = update_single_vector(index, vector_id, NAMESPACE_TO_UPDATE, TEXT_TO_ADD, METADATA_TEXT_FIELD)
        if success:
            success_count += 1
        else:
            failure_count += 1

    print("\n--- Update Summary ---")
    print(f"Successfully updated: {success_count}")
    print(f"Failed to update:   {failure_count}")
    print("------------------------")

    if vector_ids and success_count > 0:
        print("\nVerifying the update for the first vector...")
        first_id = [vector_ids[0]]
        verify_fetch = index.fetch(ids=first_id, namespace=NAMESPACE_TO_UPDATE)
        if verify_fetch.vectors:
            updated_text = verify_fetch.vectors[first_id[0]].metadata[METADATA_TEXT_FIELD]
            print(f"Updated text for vector '{first_id[0]}':")
            print(updated_text)
        else:
            print(f"Could not verify vector {first_id[0]}.")

if __name__ == "__main__":
    main()
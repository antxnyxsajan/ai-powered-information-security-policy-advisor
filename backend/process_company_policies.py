import re
import os

def preprocess_and_separate_policies(full_text):
    """
    Parses the full ISMS policy document, finds each distinct policy,
    and saves it as a separate text file in a dedicated directory.
    """
    # 1. Define the name for the output directory and create it if it doesn't exist.
    output_dir = "separated_policies"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")

    # 2. Define the regular expression pattern to find the start of each policy.
    # --- THIS IS THE CORRECTED LINE ---
    # This new, more flexible pattern looks for any line that starts with "8."
    # followed by one or two digits and a dot (e.g., "8.1." or "8.23.").
    # This correctly captures all policy variations in your document.
    pattern = re.compile(r'^(?=8\.\d{1,2}\.\s+)', re.MULTILINE)

    # 3. Split the full document text into a list of individual policy strings.
    # The first item in the list is the introductory text before the first policy, so we skip it using [1:].
    policy_chunks = pattern.split(full_text)[1:]

    if not policy_chunks:
        print("No policies found matching the expected format '8.x. ...'.")
        print("Please ensure your policy_document.txt file contains policy headings like '8.1. Policy for...'")
        return

    print(f"Found {len(policy_chunks)} policies to separate.")

    # 4. Loop through each extracted policy chunk, clean it, and save it to a file.
    for policy_text in policy_chunks:
        # Remove any leading/trailing whitespace from the policy content.
        policy_text = policy_text.strip()
        if not policy_text:
            continue

        # Extract the first line to use as the title for the filename.
        try:
            first_line = policy_text.split('\n', 1)[0].strip()
        except IndexError:
            continue # Skip if the chunk is empty

        # --- Sanitize the title to create a valid filename ---
        # a. Remove the numerical prefix (e.g., "8.1. ").
        clean_title = re.sub(r'^8\.\d{1,2}\.\s+', '', first_line)
        # b. Remove characters that are not letters, numbers, underscores, or hyphens.
        filename = re.sub(r'[^\w\s-]', '', clean_title).strip()
        # c. Replace spaces and hyphens with a single underscore, convert to lowercase, and add .txt extension.
        filename = re.sub(r'[-\s]+', '_', filename).lower() + ".txt"

        # Create the full path for the new file.
        file_path = os.path.join(output_dir, filename)

        # Write the full policy text to the new file.
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(policy_text)
            print(f"Successfully saved: {file_path}")
        except OSError as e:
            print(f"Error saving file {file_path}: {e}")


# --- How to use this script ---
# 1. Make sure your "policy_document.txt" file is in the same directory as this script.
# 2. Run this script from your terminal: python process_policies.py

if __name__ == "__main__":
    try:
        # Read the content from the source document.
        with open('policy_document.txt', 'r', encoding='utf-8') as file:
            full_document_text = file.read()
        
        # Call the main function to process the text.
        preprocess_and_separate_policies(full_document_text)

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("The file 'policy_document.txt' was not found.")
        print("Please place it in the same folder as this script before running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
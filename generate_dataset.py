# generate_dataset.py
import google.generativeai as genai
import json
import os
import time
from kaggle_secrets import UserSecretsClient

# verification of google_api_key
user_secrets = UserSecretsClient()
google_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

OUTPUT_FILE = "CustomerServ-1K.jsonl" # Start with 1K for speed, can increase later
NUM_SAMPLES_TO_GENERATE = 1000 # Your paper mentions 10K, but 1K is a good start
SAMPLES_PER_API_CALL = 10

def generate_queries():
    """
    Uses the Gemini API to generate a benchmark dataset of customer service queries.
    """
    print(f"Generating {NUM_SAMPLES_TO_GENERATE} samples...")
    
    # We will write to a JSON Lines file
    with open(OUTPUT_FILE, 'w') as f:
        for i in range(NUM_SAMPLES_TO_GENERATE // SAMPLES_PER_API_CALL):
            print(f"Generating batch {i+1}...")
            prompt = f"""
            Generate {SAMPLES_PER_API_CALL} unique, realistic e-commerce customer service queries.
            For each query, provide a JSON object with two keys:
            1. "query_text": The full text of the customer's request.
            2. "ground_truth_specialty": The correct department for the query, chosen from one of the following exact strings: ["customer returns and refunds", "technical support for software", "legal contract review"].

            RULES:
            - Ensure a balanced distribution among the three specialties.
            - Do not output anything other than the list of JSON objects.
            - The output must be a valid JSON list.

            Example:
            [
                {{"query_text": "I can't seem to reset my password.", "ground_truth_specialty": "technical support for software"}},
                {{"query_text": "My new headphones arrived with a crack, I need to send them back for a new pair.", "ground_truth_specialty": "customer returns and refunds"}}
            ]
            """
            
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                # Clean up the response to be valid JSON
                cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                queries = json.loads(cleaned_response)

                for query in queries:
                    f.write(json.dumps(query) + '\n')

            except Exception as e:
                print(f"An error occurred: {e}")
                # Simple rate limiting
                time.sleep(5)
    
    print(f"Dataset generation complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_queries()

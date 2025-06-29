# generate_dataset.py

import torch
import transformers
import json
import time
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# --- Configuration ---
OUTPUT_FILE = "CustomerServ-1K.jsonl"
NUM_SAMPLES_TO_GENERATE = 10000
SAMPLES_PER_API_CALL = 10
MODEL_ID = "google/gemma-2-2b-it" # Using the new Gemma 2 instruction-tuned model

def setup_gemma_pipeline():
    """
    Logs into Hugging Face and sets up the local Gemma 2 model for text generation.
    """
    # 1. Login to Hugging Face using the token stored in Kaggle Secrets
    try:
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HUGGING_FACE_TOKEN")
        login(token=hf_token)
        print("Successfully logged into Hugging Face.")
    except Exception as e:
        print(f"Could not log into Hugging Face. Please ensure HUGGING_FACE_TOKEN is set correctly in Kaggle Secrets. Error: {e}")
        return None

    print(f"Setting up local model: {MODEL_ID}")
    
    # 2. Configure 4-bit quantization for efficient memory usage
    quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
    
    # 3. Load the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance
        device_map="auto",
    )
    
    # 4. Create the text-generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    print("Gemma 2 pipeline is ready.")
    return pipeline

def generate_queries():
    """
    Uses a local Gemma model to generate the benchmark dataset.
    UPDATED with more robust parsing and error debugging.
    """
    pipeline = setup_gemma_pipeline()
    
    print(f"Generating {NUM_SAMPLES_TO_GENERATE} samples...")
    
    with open(OUTPUT_FILE, 'w') as f:
        num_generated = 0
        retries = 0
        max_retries = 20 # Stop after too many consecutive errors

        while num_generated < NUM_SAMPLES_TO_GENERATE and retries < max_retries:
            print(f"Generating batch... ({num_generated}/{NUM_SAMPLES_TO_GENERATE})")
            
            prompt = f"""<bos><start_of_turn>user
Generate {SAMPLES_PER_API_CALL} unique, realistic e-commerce customer service queries.
For each query, provide a JSON object with two keys:
1. "query_text": The full text of the customer's request.
2. "ground_truth_specialty": The correct department for the query, chosen from one of the following exact strings: ["customer returns and refunds", "technical support for software", "legal contract review"].

RULES:
- Ensure a balanced distribution among the three specialties.
- Your output must be ONLY a valid JSON list of objects, enclosed in ```json ... ```. Do not add any other text or explanation.
<end_of_turn>
<start_of_turn>model
```json
"""
            
            try:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                
                generated_text = outputs[0]['generated_text']

                # --- NEW ROBUST PARSING LOGIC ---
                # Find the start and end of the JSON block
                start_marker = "```json"
                end_marker = "```"
                
                start_index = generated_text.find(start_marker)
                
                # If the start marker is found, find the end marker after it
                if start_index != -1:
                    # Adjust start_index to be after the marker itself
                    start_index += len(start_marker)
                    end_index = generated_text.find(end_marker, start_index)
                    
                    if end_index != -1:
                        # Extract the JSON part
                        json_part = generated_text[start_index:end_index].strip()
                        
                        # Now, attempt to parse the extracted part
                        queries = json.loads(json_part)

                        for query in queries:
                            if num_generated < NUM_SAMPLES_TO_GENERATE:
                                f.write(json.dumps(query) + '\n')
                                num_generated += 1
                        
                        print(f"Successfully generated and wrote {len(queries)} samples.")
                        retries = 0 # Reset retry counter on success
                    else:
                        # This case means we found ```json but not the closing ```
                        raise ValueError("Found JSON start marker but no end marker.")
                else:
                    # This case means the model didn't even start with ```json
                    raise ValueError("JSON start marker not found in model output.")

            except Exception as e:
                # --- NEW DEBUGGING PRINT ---
                # This is the most important part for debugging.
                # It shows us what the model ACTUALLY generated.
                print(f"An error occurred during generation or parsing: {e}")
                print("------ MODEL'S RAW OUTPUT ------")
                if 'outputs' in locals() and outputs:
                    print(outputs[0]['generated_text'])
                else:
                    print("Model did not produce any output.")
                print("------------------------------")
                
                retries += 1
                print(f"Retry attempt {retries}/{max_retries}. Waiting a moment...")
                time.sleep(2)
    
    if num_generated < NUM_SAMPLES_TO_GENERATE:
        print(f"Halting generation due to {max_retries} consecutive errors.")
    else:
        print(f"Dataset generation complete. Saved to {OUTPUT_FILE}")
if __name__ == "__main__":
    generate_queries()

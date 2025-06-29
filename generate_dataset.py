# generate_dataset.py

import torch
import transformers
import json
import time
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# --- Configuration ---
OUTPUT_FILE = "CustomerServ-1K.jsonl"
NUM_SAMPLES_TO_GENERATE = 1000 
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
    Uses a local Gemma 2 model to generate the benchmark dataset.
    """
    pipeline = setup_gemma_pipeline()
    if pipeline is None:
        return

    print(f"Generating {NUM_SAMPLES_TO_GENERATE} samples...")
    
    with open(OUTPUT_FILE, 'w') as f:
        num_generated = 0
        while num_generated < NUM_SAMPLES_TO_GENERATE:
            print(f"Generating batch... ({num_generated}/{NUM_SAMPLES_TO_GENERATE})")
            
            # Gemma 2 uses a specific chat template.
            messages = [
                {"role": "user", "content": f"""Generate {SAMPLES_PER_API_CALL} unique, realistic e-commerce customer service queries.
For each query, provide a JSON object with two keys:
1. "query_text": The full text of the customer's request.
2. "ground_truth_specialty": The correct department for the query, chosen from one of the following exact strings: ["customer returns and refunds", "technical support for software", "legal contract review"].

RULES:
- Ensure a balanced distribution among the three specialties.
- Your output must be ONLY a valid JSON list of objects, enclosed in ```json ... ```. Do not add any other text or explanation."""},
                {"role": "model", "content": "```json\n"} # Prime the model to start with the JSON block
            ]
            
            # Apply the chat template to format the prompt correctly
            prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            try:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=1536, # Give it enough space to generate the full list
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                
                generated_text = outputs[0]['generated_text']
                
                # Extract the JSON part from the response
                json_part = generated_text.split("```json")[1].split("```")[0].strip()
                queries = json.loads(json_part)

                for query in queries:
                    if num_generated < NUM_SAMPLES_TO_GENERATE:
                        f.write(json.dumps(query) + '\n')
                        num_generated += 1
                
                print(f"Successfully generated and wrote {len(queries)} samples.")

            except Exception as e:
                print(f"An error occurred during generation or parsing: {e}")
                time.sleep(2)
    
    print(f"Dataset generation complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_queries()

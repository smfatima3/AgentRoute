import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from SafeSieve.llm.format import Message
from SafeSieve.llm.price import cost_count
from SafeSieve.llm.llm import LLM
from SafeSieve.llm.llm_registry import LLMRegistry


# Configurable parameters (it is recommended to also support loading from .env)

API_KEY = ''
API_URL = ''

# Debug Switch - Set to True to view detailed API requests and responses
DEBUG_API = os.getenv('DEBUG_API', 'False').lower() == 'true'

# Qwen-7B-instruct & deepseek - API
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = API_URL
    authorization_key = API_KEY
    headers = {
        'Content-Type': 'application/json',
        'Authorization': authorization_key
    }

    data = {
      "model": model,
      "messages": msg
    }
    
    if DEBUG_API:
        print(f"[DEBUG] API Request to {request_url}")
        print(f"[DEBUG] Model: {model}")
        print(f"[DEBUG] Messages: {msg}")
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Data: {data}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            # Add HTTP status code check
            if response.status != 200:
                error_text = await response.text()
                print(f"HTTP Error {response.status}: {error_text}")
                raise Exception(f"HTTP error {response.status}: {error_text}")
            
            try:
                response_data = await response.json()
            except Exception as e:
                error_text = await response.text()
                print(f"JSON Parse Error: {e}, Response text: {error_text}")
                raise Exception(f"Failed to parse JSON response: {e}")
            
            if DEBUG_API:
                print(f"[DEBUG] API Response status: {response.status}")
                print(f"[DEBUG] API Response data: {response_data}")
            
            # Add detailed error checking and debugging information
            if "error" in response_data:
                print(f"API Error: {response_data['error']}")
                raise Exception(f"API error: {response_data['error']}")
            
            if "choices" not in response_data:
                print(f"API Response missing 'choices' field. Full response: {response_data}")
                raise Exception(f"API response format error: missing 'choices' field. Response: {response_data}")
            
            if not response_data["choices"]:
                print(f"API Response has empty 'choices' array. Full response: {response_data}")
                raise Exception(f"API response format error: empty 'choices' array")
            
            choice = response_data["choices"][0]
            if "message" not in choice:
                print(f"API Response choice missing 'message' field. Choice: {choice}")
                raise Exception(f"API response format error: missing 'message' field in choice")
            
            if "content" not in choice["message"]:
                print(f"API Response message missing 'content' field. Message: {choice['message']}")
                raise Exception(f"API response format error: missing 'content' field in message")
                
            content = choice["message"]["content"]
            
            if isinstance(content, str):
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt, content, "gpt-3.5-turbo-instruct", model)
                return content
            else:
                print(f"API Response content is not string. Content type: {type(content)}, Content: {content}")
                raise Exception(f"API response format error: content is not a string, got {type(content)}")

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
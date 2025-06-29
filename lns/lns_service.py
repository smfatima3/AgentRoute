# /lns/lns_service.py

import redis
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time
import json

from . import config

class LocationNamingService:
    """
    Manages agent registration, status, and discovery via semantic capability matching.
    
    This service is responsible for:
    1.  Maintaining a registry of all active agents and their metadata.
    2.  Generating and storing vector embeddings of agent capabilities.
    3.  Finding agents based on the semantic similarity of their capabilities to a query.
    """

    def __init__(self):
        """
        Initializes the LNS by connecting to Redis and loading the sentence embedding model.
        """
        print("Initializing LocationNamingService...")

        # Connect to Redis
        try:
            self.redis_client = redis.StrictRedis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=False  # Set to False to handle bytes for embeddings
            )
            self.redis_client.ping()
            print("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            raise

        # Determine the best device for the model (GPU/TPU/CPU)
        # This is important for environments like Kaggle.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available(): # For Apple Silicon
             self.device = "mps"
        # Kaggle TPUs are accessed differently, often via PyTorch/XLA.
        # For simplicity, we'll default to CPU if no GPU is found.
        else:
            self.device = "cpu"
        
        print(f"Loading embedding model '{config.EMBEDDING_MODEL}' onto device: '{self.device}'")
        
        # Load the specified sentence embedding model
        # The model is loaded once during initialization to save resources.
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Helper function to generate a vector embedding for a given text.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def register_agent(self, agent_id: str, capabilities: List[str], location_context: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Registers a new agent with the LNS or updates an existing one.

        Args:
            agent_id (str): A unique identifier for the agent.
            capabilities (List[str]): A list of natural language descriptions of the agent's skills.
            location_context (str): Information about the agent's deployment location.
            metadata (Dict, optional): Any additional JSON-serializable data.

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        print(f"Registering agent: {agent_id}")
        
        # --- Step 1: Store Agent Metadata ---
        # We use a Redis Hash to store structured agent data.
        agent_meta_key = f"{config.AGENT_METADATA_PREFIX}{agent_id}"
        agent_data = {
            "capabilities": json.dumps(capabilities), # Store as JSON string
            "location_context": location_context,
            "last_heartbeat": time.time(),
            "load_factor": 0.0, # Default load factor
            "metadata": json.dumps(metadata or {})
        }
        self.redis_client.hmset(agent_meta_key, agent_data)

        # --- Step 2: Generate and Store Capability Embedding ---
        # We create a single, representative embedding for all capabilities by joining them.
        # This is a simple but effective strategy for creating a holistic agent profile.
        combined_capabilities = ". ".join(capabilities)
        embedding = self._generate_embedding(combined_capabilities)

        # Store the embedding as raw bytes in Redis.
        agent_emb_key = f"{config.AGENT_EMBEDDING_PREFIX}{agent_id}"
        self.redis_client.set(agent_emb_key, embedding.tobytes())
        
        print(f"Agent {agent_id} registered with {len(capabilities)} capabilities.")
        return True

    def update_agent_status(self, agent_id: str, load_factor: float):
        """
        Updates the status (heartbeat and load) of a registered agent.
        """
        agent_meta_key = f"{config.AGENT_METADATA_PREFIX}{agent_id}"
        if not self.redis_client.exists(agent_meta_key):
            print(f"Warning: Attempted to update status for non-existent agent: {agent_id}")
            return False
            
        update_data = {
            "last_heartbeat": time.time(),
            "load_factor": load_factor
        }
        self.redis_client.hmset(agent_meta_key, update_data)
        return True

    def find_capable_agents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Finds the most suitable agents for a given query based on semantic similarity.

        Args:
            query (str): The natural language query or task description.
            top_k (int): The number of top matching agents to return.

        Returns:
            List[Dict[str, Any]]: A sorted list of dictionaries, each containing
                                  agent_id and similarity_score.
        """
        print(f"Finding capable agents for query: '{query}'")
        
        # --- Step 1: Generate Query Embedding ---
        query_embedding = self._generate_embedding(query)

        # --- Step 2: Retrieve All Agent Embeddings ---
        # Note: For very large-scale systems (10,000+ agents), this scan can be slow.
        # A production system would use a dedicated vector search index like Redis Search,
        # FAISS, or Milvus. For this implementation, SCAN is sufficient.
        all_agent_ids = [key.decode('utf-8').split(config.AGENT_EMBEDDING_PREFIX)[1] for key in self.redis_client.scan_iter(f"{config.AGENT_EMBEDDING_PREFIX}*")]
        
        if not all_agent_ids:
            print("No agents registered in the LNS.")
            return []

        agent_embeddings_bytes = self.redis_client.mget([f"{config.AGENT_EMBEDDING_PREFIX}{agent_id}" for agent_id in all_agent_ids])
        
        # Convert bytes back to numpy arrays
        agent_embeddings = np.array([np.frombuffer(emb, dtype=np.float32) for emb in agent_embeddings_bytes])

        # --- Step 3: Calculate Cosine Similarities ---
        # The sentence-transformer utility function is highly optimized for this.
        similarities = self.model.similarity(query_embedding, agent_embeddings)[0]
        
        # --- Step 4: Rank and Return Top-K Agents ---
        # Pair agents with their scores and sort.
        results = []
        for i, agent_id in enumerate(all_agent_ids):
            results.append({
                "agent_id": agent_id,
                "similarity_score": float(similarities[i])
            })
        
        # Sort by score in descending order
        sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        
        print(f"Found {len(sorted_results)} matching agents. Returning top {top_k}.")
        return sorted_results[:top_k]

    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """Retrieves all metadata for a specific agent."""
        agent_meta_key = f"{config.AGENT_METADATA_PREFIX}{agent_id}"
        agent_data_bytes = self.redis_client.hgetall(agent_meta_key)
        
        if not agent_data_bytes:
            return None
        
        # Decode byte values to strings
        agent_data = {key.decode('utf-8'): val.decode('utf-8') for key, val in agent_data_bytes.items()}
        
        # Deserialize JSON fields
        agent_data['capabilities'] = json.loads(agent_data.get('capabilities', '[]'))
        agent_data['metadata'] = json.loads(agent_data.get('metadata', '{}'))
        
        return agent_data
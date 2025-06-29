# /lns/in_memory_lns.py

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time

from . import config

class InMemoryLNS:
    """
    An in-memory version of the LocationNamingService for testing and development
    in environments without a Redis server (like Kaggle notebooks).

    It uses the same methods and logic but stores data in Python dictionaries.
    """

    def __init__(self):
        print("Initializing InMemoryLNS (no Redis connection)...")
        # --- Data Stores ---
        self.agent_metadata = {}  # Replaces Redis Hashes for metadata
        self.agent_embeddings = {}  # Replaces Redis keys for embeddings

        # --- Model Initialization (same as the Redis version) ---
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading embedding model '{config.EMBEDDING_MODEL}' onto device: '{self.device}'")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        print("Model loaded successfully.")

    def _generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def register_agent(self, agent_id: str, capabilities: List[str], location_context: str, metadata: Dict[str, Any] = None) -> bool:
        print(f"Registering agent (in-memory): {agent_id}")
        
        # Store metadata
        self.agent_metadata[agent_id] = {
            "capabilities": capabilities,
            "location_context": location_context,
            "last_heartbeat": time.time(),
            "load_factor": 0.0,
            "metadata": metadata or {}
        }

        # Generate and store embedding
        combined_capabilities = ". ".join(capabilities)
        embedding = self._generate_embedding(combined_capabilities)
        self.agent_embeddings[agent_id] = embedding
        
        return True

    def update_agent_status(self, agent_id: str, load_factor: float):
        if agent_id not in self.agent_metadata:
            print(f"Warning: Agent not found: {agent_id}")
            return False
            
        self.agent_metadata[agent_id]["last_heartbeat"] = time.time()
        self.agent_metadata[agent_id]["load_factor"] = load_factor
        return True

    def find_capable_agents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"Finding capable agents for query (in-memory): '{query}'")
        query_embedding = self._generate_embedding(query)

        if not self.agent_embeddings:
            return []

        # Prepare embeddings for comparison
        agent_ids = list(self.agent_embeddings.keys())
        embeddings_matrix = np.array(list(self.agent_embeddings.values()))
        
        # Calculate similarities
        similarities = self.model.similarity(query_embedding, embeddings_matrix)[0]

        # Rank results
        results = [
            {"agent_id": agent_id, "similarity_score": float(score)}
            for agent_id, score in zip(agent_ids, similarities)
        ]
        
        sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_results[:top_k]

    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        return self.agent_metadata.get(agent_id)

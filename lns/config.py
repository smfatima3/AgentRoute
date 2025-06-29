# /lns/config.py

# --- Redis Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# --- Model Configuration ---
# Using the model specified by the user.
# This is a large model and may require significant RAM/VRAM.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

# --- LNS Service Configuration ---
# Redis key prefixes to avoid collisions in a shared Redis instance
AGENT_METADATA_PREFIX = "agent:meta:"
AGENT_EMBEDDING_PREFIX = "agent:emb:"
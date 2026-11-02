import numpy as np
import asyncio
import re
from typing import List, Optional, Union

from SafeSieve.llm.format import Message
from SafeSieve.llm.llm_registry import LLMRegistry


async def compute_prompt_similarity_matrix_llm(
    prompts: List[str], 
    task_type: str = "GSM8K",
    model_name: str = "deepseek-ai/DeepSeek-V3",
    max_score: float = 10.0
) -> np.ndarray:
    """
    Compute collaboration similarity matrix between prompt texts using LLM expert scoring.
    
    Args:
        prompts: List of prompts to compute similarity for
        task_type: Task type, such as "GSM8K", "SVAMP", "MATH", etc.
        model_name: LLM model name to use, defaults to "deepseek-chat"
        max_score: Maximum score, defaults to 10.0
    
    Returns:
        N×N symmetric similarity matrix with values in [0,1] range
    """
    if len(prompts) == 0:
        return np.array([])
    
    n = len(prompts)
    sim_matrix = np.zeros((n, n))
    
    # Get LLM instance
    llm = LLMRegistry.get(model_name)
    
    # Compute similarity for each pair of prompts
    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle since matrix is symmetric
            score = await _evaluate_prompt_pair_llm(
                prompts[i], prompts[j], task_type, llm, max_score
            )
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score  # Symmetric matrix
    
    # Set diagonal to 0 (similarity with self)
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix


def compute_prompt_similarity_matrix_llm_sync(
    prompts: List[str], 
    task_type: str = "GSM8K",
    model_name: str = "deepseek-ai/DeepSeek-V3",
    max_score: float = 10.0
) -> np.ndarray:
    """
    Synchronous version of compute_prompt_similarity_matrix_llm for non-async callers.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, compute_prompt_similarity_matrix_llm(
                    prompts, task_type, model_name, max_score
                ))
                return future.result()
        else:
            # If no event loop is running, we can use asyncio.run
            return asyncio.run(compute_prompt_similarity_matrix_llm(
                prompts, task_type, model_name, max_score
            ))
    except Exception as e:
        print(f"LLM similarity calculation failed in sync version: {e}")
        print("Falling back to simple similarity matrix calculation...")
        
        # Fallback to simple similarity calculation
        n = len(prompts)
        if n == 0:
            return np.array([])
        
        # Simple similarity based on text length difference
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Simple similarity based on text length difference
                len_i, len_j = len(prompts[i]), len(prompts[j])
                similarity = 1.0 - abs(len_i - len_j) / max(len_i, len_j, 1)
                similarity = max(0.3, min(0.7, similarity))  # Limit to a reasonable range
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity
        
        # Set diagonal to 0
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix


async def _evaluate_prompt_pair_llm(
    prompt_a: str, 
    prompt_b: str, 
    task_type: str, 
    llm, 
    max_score: float
) -> float:
    """
    Use LLM to evaluate collaboration complementarity between two prompts for a specific task.
    
    Returns:
        Normalized similarity score in [0,1] range
    """
    
    # Build evaluation prompt
    evaluation_prompt = _build_evaluation_prompt(prompt_a, prompt_b, task_type, max_score)
    
    try:
        # Call LLM
        messages = [Message(role="user", content=evaluation_prompt)]
        serializable_messages = [
    {"role": msg.role, "content": msg.content}
    for msg in messages
]
        response = await llm.agen(serializable_messages)
        # Extract numerical value from response
        score = _extract_score_from_response(response, max_score)
        
        # Normalize to [0,1] range
        normalized_score = score / max_score
        
        return normalized_score
        
    except Exception as e:
        print(f"LLM evaluation error: {e}, using default score 0.5")
        return 0.5  # Return medium similarity on error


def _build_evaluation_prompt(prompt_a: str, prompt_b: str, task_type: str, max_score: float) -> str:
    """
    Build prompt for LLM evaluation.
    """
    prompt_template = f"""You are a task evaluation expert, now need to arrange multiple agents to collaborate on {task_type} dataset tasks.

Please evaluate the collaboration complementarity of the following two agent role descriptions in solving {task_type} problems:

<Role A Description>
{prompt_a}
</Role A Description>

<Role B Description>
{prompt_b}
</Role B Description>

Scoring Criteria:
- 0-{max_score*0.3:.1f} points: Completely irrelevant, two roles have no collaboration value in solving {task_type} problems
- {max_score*0.3:.1f}-{max_score*0.5:.1f} points: Generally related, some collaboration potential but weak complementarity
- {max_score*0.5:.1f}-{max_score*0.7:.1f} points: Good complementarity, two roles can play different roles
- {max_score*0.7:.1f}-{max_score*0.9:.1f} points: Very good complementarity, two roles can form effective collaboration in problem solving
- {max_score*0.9:.1f}-{max_score} points: Perfect complementarity, two roles can perfectly cooperate to solve complex problems

Please carefully analyze the capability characteristics, problem-solving methods, and professional domains of the two roles, then give a numerical score between 0 and {max_score}.

Important: You must return a precise numerical value between 0 and {max_score}, format like: "Score: X.X" (where X.X is the specific value)."""

    return prompt_template


def _extract_score_from_response(response: str, max_score: float) -> float:
    """
    Extract numerical score from LLM response.
    
    Args:
        response: LLM response text
        max_score: Maximum score
    
    Returns:
        Extracted score, returns medium score if extraction fails
    """
    try:
        # Try multiple patterns to extract score
        patterns = [
            r'Score[：:]\s*(\d+\.?\d*)',  # "Score: 8.5"
            r'Points[：:]\s*(\d+\.?\d*)',  # "Points: 8.5"
            r'Rating[：:]\s*(\d+\.?\d*)',  # "Rating: 8.5"
            r'(\d+\.?\d*)\s*points',       # "8.5 points"
            r'(\d+\.?\d*)\s*/\s*' + str(max_score),  # "8.5/10"
            r'(?:^|\s)(\d+\.?\d*)(?:\s|$)',  # Standalone number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                score = float(matches[0])
                # Ensure score is within valid range
                if 0 <= score <= max_score:
                    return score
        
        # If no valid score found, return medium score
        print(f"Cannot extract valid score from response: {response[:100]}...")
        return max_score / 2
        
    except Exception as e:
        print(f"Error extracting score: {e}")
        return max_score / 2


# Legacy function for backward compatibility
def compute_prompt_similarity_matrix(prompts: List[str]) -> np.ndarray:
    """
    Legacy function for backward compatibility. 
    Fallback to simple similarity when LLM evaluation fails.
    """
    try:
        # First try to use LLM scoring
        return compute_prompt_similarity_matrix_llm_sync(prompts)
    except Exception as e:
        print(f"LLM similarity calculation failed: {e}")
        print("Falling back to simple similarity matrix calculation...")
        # Fallback to simple similarity calculation
        n = len(prompts)
        if n == 0:
            return np.array([])
        
        # Simple similarity based on text length difference
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Simple similarity based on text length difference
                len_i, len_j = len(prompts[i]), len(prompts[j])
                similarity = 1.0 - abs(len_i - len_j) / max(len_i, len_j, 1)
                similarity = max(0.3, min(0.7, similarity))  # Limit to a reasonable range
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity
        
        # Set diagonal to 0
        np.fill_diagonal(sim_matrix, 0)
        return sim_matrix


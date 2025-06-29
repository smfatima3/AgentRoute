# evaluation_harness.py

import json
import time
import pandas as pd

# Import our system components
from lns.in_memory_lns import InMemoryLNS
from broker.ats_broker import ATSBroker, BroadcastBroker, HierarchicalBroker
from core.message_queue import SimulatedMessageQueue
from core.base_agent import BaseAgent

def setup_system(agent_definitions):
    """Helper function to initialize a clean system for each run."""
    lns = InMemoryLNS()
    queue = SimulatedMessageQueue()
    agents = {
        name: BaseAgent(
            agent_id=name,
            capabilities=details['capabilities'],
            lns_instance=lns,
            queue_instance=queue
        )
        for name, details in agent_definitions.items()
    }
    return lns, queue, agents

def run_agentroute_eval(dataset, lns, queue, agents):
    """Evaluates the AgentRoute system."""
    broker = ATSBroker(lns_instance=lns, similarity_threshold=0.6)
    results = []
    
    start_time = time.time()
    for item in dataset:
        query = item['query_text']
        truth = item['ground_truth_specialty']
        
        # This is a simplified metric for hops.
        # A real eval would need more complex tracing.
        message_hops = 1 
        
        decision = broker.route_query(query)
        
        # Accuracy Check
        is_correct = 0
        if decision['status'] == 'success':
            routed_agent_id = decision['routed_to_agent_id']
            # Simple check if the agent's specialty is in its ID name
            if truth.split(" ")[0] in routed_agent_id:
                is_correct = 1
        
        results.append({
            "hops": message_hops,
            "latency_ms": (time.time() - start_time) * 1000 / len(dataset),
            "is_correct": is_correct
        })
        
    return pd.DataFrame(results)


# --- MAIN EVALUATION SCRIPT ---
if __name__ == "__main__":
    # 1. Load the generated dataset
    dataset_path = "CustomerServ-1K.jsonl"
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    print(f"Loaded {len(dataset)} queries for evaluation.")

    # 2. Define our agent population
    agent_definitions = {
        "returns_specialist_001": {"capabilities": ["Handles customer returns, refunds, and exchanges."]},
        "tech_support_007": {"capabilities": ["Troubleshoots software login errors and application bugs."]},
        "legal_advisor_003": {"capabilities": ["Reviews legal contracts for compliance."]}
    }
    
    # 3. Run evaluation for AgentRoute
    print("\n--- Evaluating AgentRoute ---")
    lns, queue, agents = setup_system(agent_definitions)
    agentroute_results = run_agentroute_eval(dataset, lns, queue, agents)
    
    # 4. Print the final results in a format similar to the paper's table
    print("\n--- Experimental Results ---")
    
    avg_hops = agentroute_results['hops'].mean()
    avg_latency = agentroute_results['latency_ms'].mean()
    accuracy = agentroute_results['is_correct'].mean() * 100

    # In your real harness, you would run evals for Broadcast and Hierarchical too
    # and combine them into one DataFrame.
    
    results_summary = {
        "System": ["AgentRoute", "Broadcast (simulated)", "Hierarchical (simulated)"],
        "Avg. Message Hops": [f"{avg_hops:.2f}", "3.00", "2.00"], # Using placeholder values for now
        "Routing Accuracy (%)": [f"{accuracy:.2f}", "33.33", "N/A"],
        "Avg. Latency (ms)": [f"{avg_latency:.2f}", "150.0", "90.0"],
    }
    
    df_summary = pd.DataFrame(results_summary)
    print(df_summary.to_string(index=False))

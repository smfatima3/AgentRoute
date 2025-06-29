# main.py

import time

from lns.in_memory_lns import InMemoryLNS
from broker.ats_broker import ATSBroker
from core.message_queue import SimulatedMessageQueue
from core.base_agent import BaseAgent

def run_demonstration():
    """
    Demonstrates the full, end-to-end workflow of the AgentRoute system.
    """
    # --- Part 1: System Initialization ---
    print("--- System Initialization ---")
    lns = InMemoryLNS()
    queue = SimulatedMessageQueue()
    broker = ATSBroker(lns_instance=lns, similarity_threshold=0.6)
    
    # --- Part 2: Agent Creation ---
    # Agents are now independent objects that register themselves with the LNS.
    print("\n--- Agent Creation ---")
    agents = [
        BaseAgent(
            agent_id="returns_specialist_001",
            capabilities=["Handles customer returns, refunds, and exchanges."],
            lns_instance=lns, queue_instance=queue
        ),
        BaseAgent(
            agent_id="tech_support_007",
            capabilities=["Troubleshoots software login errors and application bugs."],
            lns_instance=lns, queue_instance=queue
        )
    ]
    # Set a load factor for one of the agents
    lns.update_agent_status(agents[0].agent_id, load_factor=0.3)
    
    # --- Part 3: Simulating a User Request through the Broker ---
    print("\n--- Simulating User Request ---")
    user_query = "My account is locked and I can't log in."
    
    # The broker decides where to route the query
    routing_decision = broker.route_query(user_query)
    
    if routing_decision['status'] == 'success':
        # The broker now PUBLISHES a message instead of just returning a decision
        target_agent_id = routing_decision['routed_to_agent_id']
        message = {
            "type": "USER_QUERY",
            "payload": user_query,
            "timestamp": time.time()
        }
        queue.publish(topic=target_agent_id, message=message)
    else:
        print("Broker failed to route the query. Halting simulation.")
        return

    # --- Part 4: Simulating the Agent Run Loop ---
    # In a real system, agents would run in parallel threads/processes.
    # Here, we simulate this by looping and letting each agent check its messages.
    print("\n--- Simulating Agent Work Cycle (5 steps) ---")
    for i in range(5):
        print(f"\n--- Cycle {i+1} ---")
        for agent in agents:
            agent.check_for_messages()
        time.sleep(0.2) # Wait a bit between cycles

if __name__ == "__main__":
    run_demonstration()

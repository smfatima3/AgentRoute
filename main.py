# main.py

from lns.in_memory_lns import InMemoryLNS
from broker.ats_broker import ATSBroker

def run_demonstration():
    """
    A demonstration script showcasing the full AgentRoute workflow:
    1. Initialize LNS and register agents.
    2. Initialize ATSBroker with the LNS.
    3. Route queries through the broker and observe the intelligent routing.
    """
    # --- Setup Phase 1: The Location Naming Service ---
    print("--- LNS Setup ---")
    lns = InMemoryLNS()

    # Register our specialist agents
    lns.register_agent(
        agent_id="returns_specialist_001",
        capabilities=["Handles customer returns, refunds, and exchanges for products."],
        location_context="aws:us-east-1"
    )
    lns.register_agent(
        agent_id="returns_specialist_002",
        capabilities=["Specializes in processing e-commerce refunds and return merchandise authorizations (RMA)."],
        location_context="aws:us-west-2"
    )
    lns.register_agent(
        agent_id="tech_support_007",
        capabilities=["Troubleshoots software login errors and application bugs."],
        location_context="gcp:europe-west-3"
    )

    # Simulate different loads. Agent 001 is less busy than Agent 002.
    lns.update_agent_status(agent_id="returns_specialist_001", load_factor=0.2)
    lns.update_agent_status(agent_id="returns_specialist_002", load_factor=0.9)
    print("LNS setup complete. Agents are registered and status updated.")
    
    # --- Setup Phase 2: The Active Tuple Space Broker ---
    print("\n--- ATSBroker Setup ---")
    broker = ATSBroker(lns_instance=lns)
    
    # --- Execution Phase: Route a Query ---
    # This query should be classified as a "returns/refund" issue.
    # The broker must then choose between the two returns specialists.
    user_query = "My item arrived broken, I want my money back."
    
    routing_decision = broker.route_query(user_query)

    print("\n--- Final Routing Decision ---")
    if routing_decision['status'] == 'success':
        agent_id = routing_decision['routed_to_agent_id']
        load = routing_decision['details']['load_factor']
        print(f"✅ Success! Query routed to agent: {agent_id}")
        print(f"   Reason: This agent was the best semantic match with the lowest load factor ({load}).")
    else:
        print(f"❌ Failed to route query. Reason: {routing_decision['reason']}")

if __name__ == "__main__":
    run_demonstration()
# main.py

from lns.in_memory_lns import InMemoryLNS
from broker.ats_broker import ATSBroker

def run_demonstration():
    """
    A demonstration script showcasing the REFINED AgentRoute workflow.
    """
    # --- Setup Phase 1: The Location Naming Service ---
    print("--- LNS Setup ---")
    lns = InMemoryLNS()

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

    # CRITICAL: Set loads to create a clear test case.
    # The best specialist is moderately busy.
    # The second-best specialist is very busy.
    # The irrelevant tech support agent is completely free.
    lns.update_agent_status(agent_id="returns_specialist_001", load_factor=0.3)
    lns.update_agent_status(agent_id="returns_specialist_002", load_factor=0.9)
    lns.update_agent_status(agent_id="tech_support_007", load_factor=0.0)
    print("LNS setup complete. Agents are registered and status updated.")
    
    # --- Setup Phase 2: The Active Tuple Space Broker ---
    print("\n--- ATSBroker Setup ---")
    # Initialize with a similarity threshold of 0.6.
    broker = ATSBroker(lns_instance=lns, similarity_threshold=0.6)
    
    # --- Execution Phase: Route a Query ---
    user_query = "My item arrived broken, I want my money back."
    
    routing_decision = broker.route_query(user_query)

    print("\n--- Final Routing Decision ---")
    if routing_decision['status'] == 'success':
        agent_id = routing_decision['routed_to_agent_id']
        load = routing_decision['details']['load_factor']
        print(f"✅ Success! Query routed to agent: {agent_id}")
        print(f"   Reason: This agent passed the semantic threshold and had the lowest load among QUALIFIED candidates.")
    else:
        print(f"❌ Failed to route query. Reason: {routing_decision['reason']}")

if __name__ == "__main__":
    run_demonstration()

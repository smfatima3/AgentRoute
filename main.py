# main.py

from lns.lns_service import LocationNamingService
import time

def run_demonstration():
    """
    A demonstration script to showcase the LNS functionality.
    """
    try:
        # Initialize the service. This will load the model and connect to Redis.
        lns = LocationNamingService()
    except Exception as e:
        print(f"Failed to initialize LNS. Please ensure Redis is running. Error: {e}")
        return

    print("\n--- Phase 1: Registering Agents ---")
    
    # Agent 1: A specialist in handling customer returns and refunds.
    lns.register_agent(
        agent_id="returns_specialist_001",
        capabilities=[
            "Processes customer return requests for e-commerce products.",
            "Issues refunds according to company policy.",
            "Handles inquiries about return status and shipping labels."
        ],
        location_context="aws:us-east-1:e-commerce_prod",
        metadata={"team": "Customer Support", "priority_level": 2}
    )

    # Agent 2: A technical support agent for software issues.
    lns.register_agent(
        agent_id="tech_support_007",
        capabilities=[
            "Troubleshoots software installation problems.",
            "Assists users with account login and password resets.",
            "Diagnoses and resolves application bugs and errors."
        ],
        location_context="gcp:europe-west-3:saas_platform",
        metadata={"team": "Technical Operations", "expertise": "Python, Docker"}
    )
    
    # Agent 3: A legal agent specializing in contracts.
    lns.register_agent(
        agent_id="legal_advisor_003",
        capabilities=[
            "Reviews and analyzes legal contracts for compliance.",
            "Provides advice on intellectual property and data privacy laws (GDPR, CCPA).",
            "Drafts non-disclosure agreements (NDAs)."
        ],
        location_context="azure:west-us:legal_dept_secure",
        metadata={"team": "Legal", "bar_admission": "California"}
    )

    print("\n--- Phase 2: Updating Agent Status ---")
    # Simulate the tech support agent being busy.
    lns.update_agent_status(agent_id="tech_support_007", load_factor=0.85)
    print("Updated load factor for tech_support_007 to 0.85")

    # Let some time pass
    time.sleep(1)
    lns.update_agent_status(agent_id="returns_specialist_001", load_factor=0.10)
    print("Updated load factor for returns_specialist_001 to 0.10")


    print("\n--- Phase 3: Finding Capable Agents (Semantic Search) ---")
    
    # Scenario: A user has a query about getting their money back for a broken item.
    user_query = "I bought a laptop and the screen is cracked. I want to send it back and get my money back."
    
    top_agents = lns.find_capable_agents(user_query, top_k=3)

    if top_agents:
        print(f"\nTop matching agents for query: '{user_query}'")
        for agent in top_agents:
            # Retrieve full details for the top agent to show all information
            details = lns.get_agent_details(agent['agent_id'])
            print(
                f"  - Agent ID: {agent['agent_id']} "
                f"(Score: {agent['similarity_score']:.4f}) "
                f"| Location: {details['location_context']} "
                f"| Load: {details['load_factor']}"
            )
    else:
        print("Could not find any suitable agents for the query.")

if __name__ == "__main__":
    # To run this, you must have a Redis server running locally.
    # If you are using Docker, you can start one with:
    # docker run --name agentroute-redis -p 6379:6379 -d redis
    run_demonstration()
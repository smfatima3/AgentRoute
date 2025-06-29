# /broker/ats_broker.py

from typing import Dict, Any

# We can use type hinting to allow for either LNS version
from lns.lns_service import LocationNamingService
from lns.in_memory_lns import InMemoryLNS

LNS_TYPE = LocationNamingService | InMemoryLNS

class ATSBroker:
    """
    The Active Tuple Space Broker (ATSBroker).

    This component is responsible for receiving a query, classifying it,
    querying the LNS for capable agents, and then applying intelligent
    routing rules (e.g., load balancing) to select the optimal agent.
    """

    def __init__(self, lns_instance: LNS_TYPE):
        """
        Initializes the broker with a connection to a Location Naming Service.

        Args:
            lns_instance: An initialized instance of either the real or in-memory LNS.
        """
        print("Initializing ATSBroker...")
        if not hasattr(lns_instance, 'find_capable_agents'):
            raise TypeError("lns_instance must be a valid LNS with a 'find_capable_agents' method.")
        self.lns = lns_instance
        print("ATSBroker initialized successfully.")

    def _classify_query_mock(self, query: str) -> str:
        """
        A mock/simulated lightweight LLM for query classification.
        
        In a real implementation, this would call an LLM API. Here, we use
        simple keywords to simulate the classification process for development.

        Returns:
            A string representing the classified capability.
        """
        query_lower = query.lower()
        if "refund" in query_lower or "return" in query_lower or "cracked" in query_lower or "broken" in query_lower:
            return "customer returns and refunds"
        elif "login" in query_lower or "password" in query_lower or "bug" in query_lower or "error" in query_lower:
            return "technical support for software"
        elif "contract" in query_lower or "legal" in query_lower or "nda" in query_lower:
            return "legal contract review"
        else:
            return "general inquiry" # Fallback classification

    def select_optimal_agent(self, candidates: list) -> Dict[str, Any] | None:
        """
        Selects the best agent from a list of candidates based on load factor.
        This implements the "context-aware routing with load balancing".
        """
        if not candidates:
            return None

        best_agent = None
        lowest_load = float('inf')

        print(f"Selecting optimal agent from {len(candidates)} candidate(s)...")
        for candidate in candidates:
            agent_id = candidate['agent_id']
            details = self.lns.get_agent_details(agent_id)
            if details:
                load = float(details['load_factor'])
                print(f"  - Evaluating Agent: {agent_id}, Load: {load}")
                if load < lowest_load:
                    lowest_load = load
                    best_agent = details
                    best_agent['agent_id'] = agent_id # ensure agent_id is in the dict
        
        return best_agent


    def route_query(self, query: str) -> Dict[str, Any]:
        """
        The main routing pipeline for an incoming query.
        """
        print(f"\n--- Broker: New Query Received ---\nQuery: '{query}'")

        # 1. Query classification using a lightweight LLM (mocked for now)
        classified_capability = self._classify_query_mock(query)
        print(f"Broker: Query classified as -> '{classified_capability}'")

        # 2. Capability matching against agent registry via LNS
        # We find all agents that are semantically similar to the classified capability.
        capable_agents = self.lns.find_capable_agents(classified_capability, top_k=5)

        if not capable_agents:
            print("Broker: No capable agents found in LNS.")
            return {"status": "failed", "reason": "no_agents_found"}

        # 3. Context-aware routing with load balancing
        # From the list of capable agents, select the one with the lowest load.
        selected_agent = self.select_optimal_agent(capable_agents)

        if not selected_agent:
            print("Broker: Could not retrieve details for candidate agents.")
            return {"status": "failed", "reason": "agent_details_unavailable"}
            
        agent_id = selected_agent['agent_id']
        print(f"Broker: Optimal agent selected -> {agent_id} (Load: {selected_agent['load_factor']})")
        
        # 4. Response aggregation and filtering (simplified)
        # In a real system, this would involve sending the message and getting a response.
        # Here, we just return the routing decision.
        return {
            "status": "success",
            "routed_to_agent_id": agent_id,
            "details": selected_agent
        }

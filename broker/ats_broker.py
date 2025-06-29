# /broker/ats_broker.py

from typing import Dict, Any

# We can use type hinting to allow for either LNS version.
# This makes the code editor-friendly and easier to understand.
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

    def __init__(self, lns_instance: LNS_TYPE, similarity_threshold: float = 0.6):
        """
        Initializes the broker.

        Args:
            lns_instance: An initialized instance of a Location Naming Service.
            similarity_threshold (float): The minimum semantic score for an agent
                                          to be considered a valid candidate.
        """
        print("Initializing ATSBroker...")
        if not hasattr(lns_instance, 'find_capable_agents'):
            raise TypeError("lns_instance must be a valid LNS with a 'find_capable_agents' method.")
        self.lns = lns_instance
        self.similarity_threshold = similarity_threshold
        print(f"ATSBroker initialized successfully with similarity threshold: {self.similarity_threshold}")

    def _classify_query_mock(self, query: str) -> str:
        """
        A mock/simulated lightweight LLM for query classification.
        
        UPDATE: Added more keywords to make the classification more robust.
        """
        query_lower = query.lower()
        
        # Condition for returns/refunds
        if "refund" in query_lower or "return" in query_lower or "cracked" in query_lower or "broken" in query_lower:
            return "customer returns and refunds"
        
        # --- UPDATED CONDITION for technical support ---
        elif "login" in query_lower or "password" in query_lower or "bug" in query_lower or "error" in query_lower or "account" in query_lower or "locked" in query_lower:
            return "technical support for software"
            
        # Condition for legal inquiries
        elif "contract" in query_lower or "legal" in query_lower or "nda" in query_lower:
            return "legal contract review"
            
        # Fallback
        else:
            return "general inquiry"

    def select_optimal_agent(self, candidates: list) -> Dict[str, Any] | None:
        """
        Selects the best agent from a list of QUALIFIED candidates based on load factor.
        """
        if not candidates:
            return None

        best_agent = None
        lowest_load = float('inf')

        print(f"Selecting optimal agent from {len(candidates)} qualified candidate(s)...")
        for candidate in candidates:
            agent_id = candidate['agent_id']
            details = self.lns.get_agent_details(agent_id)
            if details:
                load = float(details['load_factor'])
                print(f"  - Evaluating Agent: {agent_id}, Score: {candidate['similarity_score']:.4f}, Load: {load}")
                if load < lowest_load:
                    lowest_load = load
                    best_agent = details
                    # Ensure agent_id is part of the returned dictionary
                    best_agent['agent_id'] = agent_id
        
        return best_agent

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        The main routing pipeline for an incoming query with two-stage filtering.
        """
        print(f"\n--- Broker: New Query Received ---\nQuery: '{query}'")

        # 1. Classify the query
        classified_capability = self._classify_query_mock(query)
        print(f"Broker: Query classified as -> '{classified_capability}'")

        # 2. Find all potentially relevant agents from LNS
        all_candidates = self.lns.find_capable_agents(classified_capability, top_k=5)

        if not all_candidates:
            print("Broker: No agents found in LNS for this classification.")
            return {"status": "failed", "reason": "no_agents_found"}

        # --- REFINED LOGIC: Step 3 - Semantic Filtering ---
        print(f"Broker: Applying semantic similarity threshold > {self.similarity_threshold}")
        qualified_candidates = []
        for agent in all_candidates:
            if agent['similarity_score'] >= self.similarity_threshold:
                qualified_candidates.append(agent)
                print(f"  - ✅ Kept: {agent['agent_id']} (Score: {agent['similarity_score']:.4f})")
            else:
                print(f"  - ❌ Discarded: {agent['agent_id']} (Score: {agent['similarity_score']:.4f} is too low)")
        
        if not qualified_candidates:
            print("Broker: No agents passed the similarity threshold.")
            return {"status": "failed", "reason": "no_sufficiently_similar_agents"}

        # --- Step 4: Load Balancing (on the qualified pool only) ---
        selected_agent = self.select_optimal_agent(qualified_candidates)

        if not selected_agent:
            print("Broker: Could not retrieve details for candidate agents.")
            return {"status": "failed", "reason": "agent_details_unavailable"}
            
        agent_id = selected_agent['agent_id']
        print(f"Broker: Optimal agent selected -> {agent_id} (Load: {selected_agent['load_factor']})")
        
        # 5. Return the final decision
        return {
            "status": "success",
            "routed_to_agent_id": agent_id,
            "details": selected_agent
        }

# /core/base_agent.py

import time
from typing import List, Dict

# Import components the agent needs to interact with
from lns.in_memory_lns import InMemoryLNS
from core.message_queue import SimulatedMessageQueue

LNS_TYPE = InMemoryLNS # Or the real one later
QUEUE_TYPE = SimulatedMessageQueue # Or a real one later

class BaseAgent:
    """
    A base class for all specialist agents in the AgentRoute system.
    """
    def __init__(self, agent_id: str, capabilities: List[str],
                 lns_instance: LNS_TYPE, queue_instance: QUEUE_TYPE):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.lns = lns_instance
        self.queue = queue_instance
        self.is_running = True
        
        # Agent registers itself with the LNS upon creation
        self.lns.register_agent(
            agent_id=self.agent_id,
            capabilities=self.capabilities,
            location_context="simulated:local"
        )
        print(f"AGENT [{self.agent_id}]: Initialized and registered with LNS.")

    def _process_message(self, message: Dict):
        """
        The core logic for what an agent does when it receives a message.
        Subclasses would override this with specific behaviors.
        """
        print(f"AGENT [{self.agent_id}]: Received and processed message of type '{message['type']}'")
        # Simulate doing some work
        time.sleep(0.5)
        # In a real system, the agent might publish a response back to the queue
        
    def check_for_messages(self):
        """
        A single step of the agent's run loop. It checks its inbox and processes messages.
        """
        # print(f"AGENT [{self.agent_id}]: Checking for messages...")
        messages = self.queue.subscribe(topic=self.agent_id)
        if messages:
            print(f"AGENT [{self.agent_id}]: Found {len(messages)} message(s) in inbox.")
            for msg in messages:
                self._process_message(msg)
        
    def stop(self):
        """Stops the agent's running loop."""
        self.is_running = False

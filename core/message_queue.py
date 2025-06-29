# /core/message_queue.py

from collections import defaultdict
from typing import Dict, List

class SimulatedMessageQueue:
    """
    A simple in-memory message queue to simulate a pub/sub system like RabbitMQ.
    This avoids external dependencies for development and testing.
    """
    def __init__(self):
        print("Initializing SimulatedMessageQueue...")
        # The queue is a dictionary where keys are topics (agent_ids)
        # and values are lists of messages.
        self._queues: Dict[str, List[Dict]] = defaultdict(list)

    def publish(self, topic: str, message: Dict):
        """
        Publishes a message to a specific topic (agent's inbox).
        """
        print(f"QUEUE: Message published to topic '{topic}'. Content: {message['type']}")
        self._queues[topic].append(message)

    def subscribe(self, topic: str) -> List[Dict]:
        """
        Retrieves all messages for a topic and clears the queue for that topic.
        This simulates message consumption.
        """
        if topic in self._queues:
            messages = self._queues[topic]
            # Clear the queue for this topic after consumption
            self._queues[topic] = []
            return messages
        return []

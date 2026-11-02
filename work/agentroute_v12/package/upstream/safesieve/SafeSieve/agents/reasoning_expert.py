from typing import List, Any, Dict
import re
import asyncio

from SafeSieve.graph.node import Node
from SafeSieve.agents.agent_registry import AgentRegistry
from SafeSieve.llm.llm_registry import LLMRegistry
from SafeSieve.prompt.prompt_set_registry import PromptSetRegistry
from SafeSieve.tools.coding.python_executor import execute_code_get_return


@AgentRegistry.register('ReasoningExpert')
class ReasoningExpert(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "ReasoningExpert", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        
    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """Process inputs for mathematical reasoning - optimized for efficiency"""
        system_prompt = self.constraint
        
        # Build the main prompt with the mathematical problem
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        
        # Add collaborative guidance prompt, refer to math_solver's prompt style
        if spatial_info:
            spatial_str = "At the same time, there are the following responses to the same question for your reference:\n\n"
            for id, info in spatial_info.items():
                agent_role = info.get('role', 'Unknown')
                agent_output = info.get('output', '')
                
                # Extract key information, avoid lengthy output
                if "\\boxed{" in agent_output:
                    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
                    matches = re.findall(boxed_pattern, agent_output)
                    if matches:
                        spatial_str += f"Agent {id} as a {agent_role} computed answer: \\boxed{{{matches[-1]}}}\n"
                        spatial_str += f"Full solution:\n{agent_output}\n\n"
                else:
                    # Only take the first 150 characters, avoid too long output
                    summary = agent_output[:150] + "..." if len(agent_output) > 150 else agent_output
                    spatial_str += f"Agent {id} as a {agent_role} provides:\n{summary}\n\n"
            
            user_prompt += spatial_str
        
        # Add temporal information, refer to math_solver's prompt style
        if temporal_info:
            temporal_str = "In the last round of dialogue, there were the following responses to the same question for your reference:\n\n"
            for id, info in temporal_info.items():
                agent_role = info.get('role', 'Unknown')
                agent_output = info.get('output', '')
                summary = agent_output[:120] + "..." if len(agent_output) > 120 else agent_output
                temporal_str += f"Previous Agent {id} as a {agent_role}:\n{summary}\n\n"
            
            user_prompt += temporal_str
            
        # Add collaborative guidance based on role
        if self.role == "Problem Decomposer & Strategist":
            user_prompt += "\nBased on the above information, provide a clear decomposition of the problem and a high-level strategy. Your output will guide other experts."
        elif self.role == "Step-by-step Executor":
            user_prompt += "\nBased on the provided strategy and any other information, execute the mathematical solution with rigorous, step-by-step detail."
        elif self.role == "Computational Verifier":
            user_prompt += "\nReview the reasoning and calculations above. Write and execute Python code to verify the results or perform complex computations. Explain your verification process and conclusions."
        elif self.role == "Answer Consolidator & Reviewer":
            user_prompt += "\nCarefully review all proposed strategies, solutions, and verifications. Synthesize them into a final, correct, and coherent answer."
            
        return system_prompt, user_prompt
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from the response"""
        code_pattern = r'```python\n(.*?)\n```'
        return re.findall(code_pattern, text, re.DOTALL)
    
    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """Execute mathematical reasoning asynchronously - optimized"""
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        
        # Get response from LLM
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        
        # Only execute code when necessary
        code_blocks = self.extract_code_blocks(response)
        if code_blocks and self.role == "Computational Verifier":
            for code in code_blocks:
                try:
                    result = execute_code_get_return(code)
                    if result and result != "Error occurred" and "Error" not in str(result):
                        response += f"\nCode result: {result}"
                except Exception as e:
                    response += f"\nCode failed: {str(e)}"
        
        return response
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """Synchronous execution wrapper - optimized"""
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        
        # Only execute code when necessary
        code_blocks = self.extract_code_blocks(response)
        if code_blocks and self.role == "Computational Verifier":
            for code in code_blocks:
                try:
                    result = execute_code_get_return(code)
                    if result and result != "Error occurred" and "Error" not in str(result):
                        response += f"\nCode result: {result}"
                except Exception as e:
                    response += f"\nCode failed: {str(e)}"
        
        return response

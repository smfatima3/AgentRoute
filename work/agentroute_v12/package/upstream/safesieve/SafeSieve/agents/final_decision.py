from typing import List,Any,Dict

from SafeSieve.graph.node import Node
from SafeSieve.agents.agent_registry import AgentRegistry
from SafeSieve.llm.llm_registry import LLMRegistry
from SafeSieve.prompt.prompt_set_registry import PromptSetRegistry
from SafeSieve.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        
        # Improved system prompt, more explicitly requires outputting only code
        system_prompt = (
            "You are a code generation AI. You MUST respond with ONLY Python code. "
            "Do not include any explanations, comments about the code quality, or text before/after the code. "
            "Your entire response should be valid Python code that can be executed directly."
        )
        
        spatial_str = ""
        best_code = None
        best_feedback = None
        
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):  # is python code
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                
                # If code that passes the tests is found, use it directly
                if is_solved and len(self.internal_tests):
                    best_code = output
                    best_feedback = "Code passes all tests"
                
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        
        # If there is code that passes the tests, create a more direct prompt
        if best_code:
            user_prompt = (
                f"The following code successfully solves the task and passes all tests:\n\n"
                f"```python\n{best_code}\n```\n\n"
                f"Output this exact code without any modifications or explanations."
            )
        else:
            user_prompt = (
                f"Task:\n{raw_inputs['task']}\n\n"
                f"Based on the following agent outputs, write the final Python code solution. "
                f"Output ONLY the Python code, no explanations:\n\n{spatial_str}"
            )
        
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        
        # Try to clean up the response, removing possible extra text
        if '```python' in response:
            # 提取代码块
            import re
            code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
        
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        
        # Try to clean up the response, removing possible extra text
        if '```python' in response:
            # 提取代码块
            import re
            code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
        
        return response


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

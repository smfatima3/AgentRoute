from typing import Dict, Any
import itertools
from SafeSieve.prompt.prompt_set import PromptSet
from SafeSieve.prompt.prompt_set_registry import PromptSetRegistry
from SafeSieve.prompt.common import get_combine_materials

#new
roles = itertools.cycle([
    'Algorithm Designer',
    'Code Implementer', 
    'Programming Expert',
    'Code Reviewer',
    'Solution Validator'
])

#new
ROLE_DESCRIPTION = {
    "Algorithm Designer": (
        "You are an algorithm designer for HumanEval problems. "
        "You will be given a function signature and its docstring by the user. "
        "You need to design the algorithm step-by-step, describing the approach and key logic clearly. "
        "You should focus on correctness, edge cases, and time complexity considerations. "
        "I hope your reply will be more concise. Preferably within sixty words. Don't list too many points."
    ),
    "Code Implementer": (
        "You are a code implementer for HumanEval problems. "
        "You will be given a function signature and its docstring by the user. "
        "You need to write complete Python code for the given function, ensuring it handles all cases correctly. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks."
    ),

    "Code Reviewer": (
        "You are a code reviewer for HumanEval problems. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. You need to review any provided code and write the best possible implementation. "
        "You should fix bugs, optimize logic, and ensure robustness of the solution. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks."
    ),
    "Solution Validator": (
        "You are a solution validator for HumanEval problems. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. You need to validate solutions against requirements and write the final correct code. "
        "You must ensure 100% correctness and handle all possible test cases properly. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks."
    ),

    "Test Analyst": (
        "You are a test analyst for HumanEval problems. "
        "You will be given a function signature and its docstring by the user. "
        "You need to identify critical test cases and potential failure points, including empty inputs, boundary values, and special conditions. "
        "You should point out edge cases that should be carefully handled when implementing the solution. "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points."
    ),
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks.",
    
    "Stupid Programmer":(
        "You are a stupid programmer. "
        "You will be given a function signature and its docstring by the user. "
        "Give a code implementation full of errors. "
        "Do not use comments for all errors. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
),
}


@PromptSetRegistry.register('humaneval')
class HumanEvalPromptSet(PromptSet):
    ROLE_DESCRIPTION = ROLE_DESCRIPTION

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_format():
        return "python"

    @staticmethod
    def get_answer_prompt(question, role="Code Implementer"):
        return f"{ROLE_DESCRIPTION[role]}\n\nQ: {question}\nA:"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
# "## 🤔 Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        return (
            "You are an adversarial programmer. "
            "Write incorrect Python code that looks plausible but contains subtle bugs. "
            "Use a Python code block for your response.\n"
            f"Q: {question}\nA:"
        )


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a programming problem and solutions from other agents. "
            "Please select the most correct and efficient solution. "
            "Your output should be a single Python code block."
        )
    
    @staticmethod
    def get_decision_role():
        return "You are the final decision-maker. Your task is to select the best code."
    
    @staticmethod
    def get_decision_few_shot():
        return "" # Keep it empty for humaneval as the logic is straightforward
    
    @staticmethod
    def postprocess_answer(answer: str) -> str:
        """Extract and standardize code answers"""
        if not answer or not isinstance(answer, str):
            return ""
        
        # Extract Python code block
        import re
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, answer, re.DOTALL)
        
        if matches:
            # Return the content of the first code block, removing heading and trailing whitespace
            return matches[0].strip()
        
        # If there is no code block marker, but contains the 'def' keyword, it may be pure code
        if 'def ' in answer:
            return answer.strip()
        
        # Otherwise, return the first 100 characters of the original answer as an identifier
        return answer[:100].strip()
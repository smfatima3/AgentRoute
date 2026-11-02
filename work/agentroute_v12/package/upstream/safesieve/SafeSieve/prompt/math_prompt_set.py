from typing import Dict, Any
import itertools
from SafeSieve.prompt.prompt_set import PromptSet
from SafeSieve.prompt.prompt_set_registry import PromptSetRegistry
from SafeSieve.prompt.common import get_combine_materials

# Optimized 4-roles-loop for MATH dataset
roles = itertools.cycle([
    'Problem Decomposer & Strategist',
    'Step-by-step Executor', 
    # 'Computational Verifier',
    'Answer Consolidator & Reviewer',
])

# Role descriptions optimized for MATH dataset
ROLE_DESCRIPTION = {
    "Problem Decomposer & Strategist": 
        "You are an expert problem solver specializing in the MATH dataset. Your task is to analyze the problem, decompose it into logical sub-steps, and create a clear strategy. "
        "Explicitly state the mathematical field (e.g., Algebra, Number Theory, etc.). "
        "Present your final strategy in a \\boxed{}. The strategy should be a high-level plan, not the full solution.",
    
    "Step-by-step Executor":
        "You are an expert mathematical reasoner focused on the MATH dataset. Your task is to execute the provided strategy step-by-step. "
        "Show all your work with rigorous, clear, and detailed reasoning for each step. "
        "Do not skip any calculations. Present your final numerical answer in \\boxed{}.",
    
    "Answer Consolidator & Reviewer":
        "You are an expert solution reviewer for the MATH dataset. Your task is to critically review the entire solution process, including the strategy, execution, and verification. "
        "Check for logical flaws, calculation errors, and ensure the final answer is consistent and correct. "
        "Synthesize all correct parts into a single, coherent, final solution. Present the final, verified answer in \\boxed{}."
}

# Few-shot examples of optimizations for MATH dataset
FEW_SHOT_DATA = {
"Problem Decomposer & Strategist":
"""Q: Solve for $x$: $4^x - 2^{x+1} - 3 = 0$.

A: **Field**: Algebra
**Analysis**: This is a quadratic-form equation. Let $y = 2^x$. The equation becomes $y^2 - 2y - 3 = 0$.
**Strategy**: 1. Substitute $y = 2^x$. 2. Solve the quadratic equation for $y$. 3. Substitute back to solve for $x$. 4. Discard any invalid solutions.
\\boxed{\\text{Solve } y^2 - 2y - 3 = 0 \\text{ where } y=2^x}""",

"Step-by-step Executor":
"""Q: Solve for $x$: $4^x - 2^{x+1} - 3 = 0$. Strategy: Solve $y^2 - 2y - 3 = 0$ where $y=2^x$.

A: **Step 1: Substitute and form quadratic**
Let $y = 2^x$. The equation is $(2^2)^x - 2 \cdot 2^x - 3 = 0$, which is $y^2 - 2y - 3 = 0$.

**Step 2: Solve the quadratic**
Factoring the quadratic: $(y-3)(y+1) = 0$.
So, $y=3$ or $y=-1$.

**Step 3: Substitute back and solve for x**
Case 1: $y=3 \implies 2^x = 3 \implies x = \log_2(3)$.
Case 2: $y=-1 \implies 2^x = -1$. This is not possible for real $x$ since $2^x > 0$.

**Step 4: Final Answer**
The only solution is $x = \log_2(3)$.

\\boxed{\\log_2(3)}""",


"Answer Consolidator & Reviewer":
"""Q: Review the solution for $4^x - 2^{x+1} - 3 = 0$.

A: **Review**:
- **Strategy**: Correctly identified the quadratic form. The plan is sound.
- **Execution**: Steps are logical. Factoring is correct. Correctly identified $2^x = -1$ as having no real solution.
- **Verification**: The computational check confirms the algebraic solution.

**Conclusion**: The solution is logically sound and computationally verified. All steps are correct.
The final answer is $x = \log_2(3)$.
\\boxed{\\log_2(3)}"""
}

@PromptSetRegistry.register('math')
class MATHPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_format():
        return "natural language with mathematical notation"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Solver"):
        # Simplified prompt format
        return f"{FEW_SHOT_DATA[role]}\n\nQ: {question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You are a mathematical arbitrator. Review all agent solutions and select the best answer. "
        "If solutions agree, present the consensus. If they differ, choose the most mathematically sound. "
        "Present final answer in \\boxed{}.")
    
    @staticmethod
    def get_decision_role():
        return "You are the mathematical arbitrator."
    
    @staticmethod
    def get_decision_few_shot():
        return """
Q: Solve $|x-3| + |x+2| = 7$

Agent Summary:
- Planner: Case analysis based on critical points -2, 3
- Solver: Found x = -3, x = 4
- Reviewer: Verified both solutions

A: All agents agree on case analysis approach. Two solutions verified:
\\boxed{x = -3 \\text{ or } x = 4}
"""
    
    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Previous solution failed:
Problem: {question}
Solution: {solution}
Feedback: {feedback}

Solve again with corrections:
{question}"""

    @staticmethod
    def get_query_prompt(question):
        return f"Analyze if additional information is needed for: {question}"

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return f"Extract mathematical information from document for: {query}\n\nDocument: {file}"

    @staticmethod
    def get_websearch_prompt(question, query):
        return f"Generate search queries for: {query}\nProblem: {question}"

    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return f"Summarize search results for: {query}\nProblem: {question}\nResults: {results}"

    @staticmethod
    def get_reflect_prompt(question, answer):
        return f"Reflect on your solution:\nProblem: {question}\nSolution: {answer}"

    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(answers)])
        return f"Select best answer from:\nProblem: {question}\n{formatted_answers}\nConstraint: {constraint}"

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(answers)])
        return f"Select best answer from:\nProblem: {question}\n{formatted_answers}\nConstraint: {constraint}"

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

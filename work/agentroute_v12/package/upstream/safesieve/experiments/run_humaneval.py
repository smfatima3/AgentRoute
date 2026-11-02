import sys
import os
import argparse
import yaml
import json
import time
import asyncio
import math
from pathlib import Path
import torch
import numpy as np
import copy
import re
from typing import List, Union, Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from SafeSieve.graph.graph import Graph
from SafeSieve.tools.reader.readers import JSONLReader
from SafeSieve.tools.coding.python_executor import PyExecutor
from SafeSieve.utils.globals import Time, Cost, PromptTokens, CompletionTokens, RoleModelTokens
from SafeSieve.utils.const import SafeSieve_ROOT
from SafeSieve.utils.utils import nuclear_norm
from SafeSieve.graph.prompt_similarity import compute_prompt_similarity_matrix, compute_prompt_similarity_matrix_llm_sync
from SafeSieve.prompt.humaneval_prompt_set import HumanEvalPromptSet
from SafeSieve.graph.edge_matrix import EdgeMatrixManager
from SafeSieve.graph.pruning_algorithm import dynamic_threshold

def load_result(result_file: str) -> List[dict]:
    path = Path(result_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_log(log_file: str) -> List[dict]:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_log(log_file: str, log_data: List[dict]):
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

def dataloader(data_list: List, batch_size: int, i_batch: int) -> List:
    start = i_batch * batch_size
    return data_list[start:start + batch_size]
    
def extract_python_code(text: str) -> str:
    code_pattern = r'```python\n(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()

    if 'def ' in text:
        return text.strip()

    return ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSieve Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--llm_name", type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument('--mode', type=str, default='FullConnected', choices=['DirectAnswer','FullConnected','Random','Chain','Debate','Layered','Star'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--imp_per_iterations', type=int, default=3)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--pruning_rate', type=float, default=0.2)
    parser.add_argument('--domain', type=str, default="humaneval")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5])
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--min_agents', type=int, default=3)
    parser.add_argument('--max_data', type=int, default=0)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--use_cost_log', action='store_true')
    
    # Pruning Parameters (only spatial now)
    parser.add_argument('--start_pruning_batch', type=int, default=5)
    parser.add_argument('--max_pruning_times', type=int, default=7)
    parser.add_argument('--min_threshold', type=float, default=0.65)
    parser.add_argument('--max_threshold', type=float, default=0.9)
    parser.add_argument('--alpha_weight', type=float, default=0.5)
    parser.add_argument('--beta_growth_rate', type=float, default=1)
    # LLM similarity calculation parameters
    parser.add_argument('--use_llm_similarity', action='store_true', help='Use LLM expert scoring for similarity calculation')
    parser.add_argument('--similarity_model', type=str, default='deepseek-ai/DeepSeek-V3', help='LLM model for similarity calculation')
    parser.add_argument('--similarity_max_score', type=float, default=10.0, help='Maximum score for LLM similarity evaluation')
    
    # Heterogeneous LLM experiment parameters
    parser.add_argument('--heterogeneous_llms', action='store_true', help='Enable heterogeneous LLM experiment')
    parser.add_argument('--llm_config', nargs='+', type=str, default=["deepseek-ai/DeepSeek-V3","gpt-4o-mini","LoRA/Qwen/Qwen2.5-72B-Instruct","@cf/meta/llama-3.1-8b-instruct","moonshotai/Kimi-K2-Instruct"], help='List of LLM models for heterogeneous experiment, e.g., deepseek-ai/DeepSeek-V3 Qwen/Qwen2.5-7B-Instruct')

    return parser.parse_args()

async def safe_arun_with_retry(graph: Graph, input_dict: dict, num_rounds: int, max_retries: int = 5, delay_seconds: int = 30):
    for attempt in range(1, max_retries + 1):
        try:
            return await graph.arun(input_dict, num_rounds)
        except KeyError as e:
            print(f"[Retry {attempt}/{max_retries}] KeyError: {e}. Retrying in {delay_seconds}s...")
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Error: {e}. Retrying in {delay_seconds}s...")
        if attempt < max_retries:
            await asyncio.sleep(delay_seconds)
    print(f"Exceeded retries, exiting.")
    sys.exit(1)

async def main():
    Cost.instance().value = 0
    PromptTokens.instance().value = 0
    CompletionTokens.instance().value = 0
    
    args = parse_args()
    RoleModelTokens.instance().value = {name:0 for name in args.llm_config}
    print("model token:", RoleModelTokens.instance().value)
    print(f">>> Starting, dataset_json={args.dataset_json}, llm={args.llm_name}")

    dataset = JSONLReader.parse_file(args.dataset_json)
    if args.max_data > 0:
        dataset = dataset[:args.max_data]
    print(f">>> Loaded {len(dataset)} items, batch_size={args.batch_size}")

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{SafeSieve_ROOT}/result/{args.domain}")
    safe_llm_name = args.llm_name.replace("/", "_")
    result_file = result_dir / f"{args.domain}_{safe_llm_name}_{current_time}.json"

    ckpt_dir = result_dir / "checkpoint"
    log_dir = result_dir / "cost_log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{args.domain}_{safe_llm_name}_{current_time}.json"
    log_file = log_dir / f"{args.domain}_{safe_llm_name}_{current_time}.json"

    if args.use_checkpoint:
        files = sorted(ckpt_dir.glob(f"{args.domain}_{safe_llm_name}_*.json"), key=os.path.getmtime, reverse=True)
        if files:
            ckpt_file = files[0]
            current_time = "_".join(ckpt_file.stem.split("_")[2:])
            result_file = result_dir / f"{args.domain}_{safe_llm_name}_{current_time}.json"
            log_file = log_dir / f"{args.domain}_{safe_llm_name}_{current_time}.json"
    if args.use_cost_log:
        files = sorted(log_dir.glob(f"{args.domain}_{safe_llm_name}_*.json"), key=os.path.getmtime, reverse=True)
        if files:
            log_file = files[0]
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    roles = list(HumanEvalPromptSet.ROLE_DESCRIPTION.keys())
    random.shuffle(roles)
    roles = roles[:len(agent_names)]
    role_prompts = [HumanEvalPromptSet.ROLE_DESCRIPTION[role] for role in roles]
    # Compute prompt similarity matrix
    if args.use_llm_similarity:
        print(f">>> Using LLM expert scoring for similarity calculation (model: {args.similarity_model})")
        similarity_matrix = compute_prompt_similarity_matrix_llm_sync(
            role_prompts, 
            task_type="HumanEval",
            model_name=args.similarity_model,
            max_score=args.similarity_max_score
        )
    else:
        print(">>> Using legacy TF-IDF similarity calculation")
        similarity_matrix = compute_prompt_similarity_matrix(role_prompts)
    
    # Print initial similarity matrix
    print(">>> Initial Similarity Matrix:")
    print(similarity_matrix)
    print()

    # Heterogeneous LLM configuration
    if args.heterogeneous_llms:
        if not args.llm_config:
            raise ValueError("--llm_config must be provided when --heterogeneous_llms is enabled")
        print(f"[Heterogeneous LLM] Enabled. LLM config: {args.llm_config}")
        print("[Heterogeneous LLM] Agent LLM assignment:")
        for i, agent_name in enumerate(agent_names):
            assigned_llm = args.llm_config[i % len(args.llm_config)]
            print(f"    Agent {i}: {agent_name} -> {assigned_llm}")
    else:
        print(f"[Homogeneous LLM] All agents use: {args.llm_name}")
        
    N = len(agent_names)
    spatial_logits = torch.full((N*N,), 0.5, dtype=torch.float32)
    kwargs = get_kwargs(args.mode, N)
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=args.decision_method,
                  optimized_spatial=args.optimized_spatial,
                  heterogeneous_llms=args.heterogeneous_llms,
                  llm_config=args.llm_config if args.heterogeneous_llms else None,
                  **kwargs)
    
    node_id_to_role = {node_id: role for node_id, role in zip(graph.node_order, roles)}
    print(">>> Agent to Role Mapping:")
    for idx, (node_id, role) in enumerate(zip(graph.node_order, roles)):
        llm = args.llm_config[idx % len(args.llm_config)] if args.heterogeneous_llms else args.llm_name
        print(f"    {node_id} -- {role} (LLM: {llm})")

    graph.spatial_logits = torch.nn.Parameter(spatial_logits, requires_grad=True)
    optimizer = torch.optim.Adam([graph.spatial_logits], lr=args.lr)
    spatial_edge_manager = EdgeMatrixManager(N, similarity_matrix, alpha=args.alpha_weight, beta=1.0, epsilon=1e-6)

    start_batch = 0
    pruning_count = 0
    log_data = load_log(log_file)
    if ckpt_file.exists():
        ckpt = json.loads(ckpt_file.read_text(encoding='utf-8'))
        start_batch = ckpt.get('batch_index', 0)
        pruning_count = ckpt.get('pruning_count', 0)
        graph.spatial_masks = torch.tensor(ckpt['spatial_masks'], dtype=torch.float32)
        print(f">>> Resuming from batch {start_batch}, pruning_count: {pruning_count}")

    num_batches = math.ceil(len(dataset)/args.batch_size)
    if start_batch >= num_batches:
        print(f">>> Checkpoint batch {start_batch} >= total batches {num_batches}, starting from beginning")
        start_batch = 0
        if ckpt_file.exists():
            ckpt_file.unlink()

    total_solved, total_executed = (0, 0)
    for i_batch in range(start_batch, num_batches):
        print(f"Batch {i_batch}", '-'*80)
        start_ts = time.time()
        tasks, add_losses, tests = [], [], []

        batch = dataloader(dataset, args.batch_size, i_batch)
        if not batch:
            break
        
        N = graph.num_nodes
        if spatial_edge_manager.N != N:
            print(f"Updating spatial_edge_manager N from {spatial_edge_manager.N} to {N}")
            spatial_edge_manager.update_matrix_size(N)

        realized_list = []
        for rec in batch:
            realized = copy.deepcopy(graph)
            realized_list.append(realized)
            realized.spatial_logits = graph.spatial_logits
            S = realized.spatial_logits.reshape((N,N))
            reg_loss = nuclear_norm(S)
            tests.append(rec['test'])
            tasks.append(asyncio.create_task(safe_arun_with_retry(realized, {'task': rec['prompt']}, args.num_rounds)))
            add_losses.append(reg_loss)

        raw_results = await asyncio.gather(*tasks)
        raw_answers, log_probs = zip(*raw_results)
        data = load_result(result_file)
        losses, utilities = [], []
               
        for task_data, answer, log_prob, add_loss, test, solved_task in zip(batch, raw_answers, log_probs, add_losses, tests, realized_list):
            if not isinstance(answer, list) or not answer:
                answer_code = ""
            else:
                answer_code = extract_python_code(answer[0])
            
            is_solved, _, _ = PyExecutor().execute(answer_code, [test], timeout=100)
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            utilities.append(is_solved)
            losses.append(-log_prob * is_solved + add_loss)

            if is_solved:
                adj = solved_task.spatial_adj_matrix
                spatial_edge_manager.update_complementarity(adj, solved=True)

            updated_item = {
                "Question": task_data['prompt'],
                "Tests": test,
                "Attempt answer": answer_code,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
            print(f"Final Log: {json.dumps(updated_item, ensure_ascii=False)}")

        adj = graph.spatial_adj_matrix
        batch_solved = any(entry['Solved'] for entry in data[-len(batch):])
        spatial_edge_manager.update_complementarity(adj, batch_solved)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        spat = graph.spatial_masks
        print(f"\nBatch {i_batch} Current Mask Matrix:")
        print("Spatial mask matrix:")
        print(graph.spatial_masks.reshape((N, N)))
        print(f"Cost: {Cost.instance().value} | PromptTokens: {PromptTokens.instance().value} | CompletionTokens: {CompletionTokens.instance().value} \nRoleModelTokens: {RoleModelTokens.instance().value}")

        dynamic_beta = 1.0 + (i_batch * args.batch_size * args.beta_growth_rate) / 100.0
        spatial_edge_manager.update_beta_weight(dynamic_beta)
        print(f"Dynamic beta weight: {dynamic_beta:.2f}")
        
        spatial_edge_score, _ = spatial_edge_manager.get_edge_score_with_debug()
        print("Spatial edge_score matrix (for pruning):")
        print(np.round(spatial_edge_score, 4))

        current_threshold = dynamic_threshold(
            beta=dynamic_beta,
            base_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            growth_rate=args.beta_growth_rate
        )
        print(f"Dynamic edge score threshold for this batch: {current_threshold:.4f}")

        should_prune = False
        spatial_below_threshold_count = 0
        active_edge_scores = np.array([])  # Initialize as empty array
        
        if ((i_batch + 1) % args.imp_per_iterations == 0) and (i_batch + 1) >= args.start_pruning_batch and pruning_count < args.max_pruning_times and args.optimized_spatial:
            # Check spatial pruning condition
            spatial_should_prune = False
            if args.optimized_spatial:
                current_mask_2d = graph.spatial_masks.reshape((N, N)).cpu().numpy()
                active_non_diagonal_mask = (current_mask_2d > 0) & (~np.eye(N, dtype=bool))
                active_edge_scores = spatial_edge_score[active_non_diagonal_mask]
                
                spatial_below_threshold_count = 0
                if active_edge_scores.size > 0:
                    spatial_below_threshold_count = np.sum(active_edge_scores < current_threshold)
                
                if spatial_below_threshold_count > 0:
                    spatial_should_prune = True
            
            should_prune = spatial_should_prune

            print(f"\nBatch {i_batch} Pruning Analysis:")
            print(f"Edge score threshold: {current_threshold:.4f}")
            if args.optimized_spatial:
                print(f"Spatial: Active non-diagonal edges below threshold: {spatial_below_threshold_count}/{active_edge_scores.size}")
            print(f"Pruning count: {pruning_count}/{args.max_pruning_times}")
            
            if should_prune:
                prune_reasons = []
                if spatial_should_prune:
                    prune_reasons.append(f"spatial ({spatial_below_threshold_count} edges below threshold)")
                print(f"Pruning condition met: {', '.join(prune_reasons)}")
            else:
                print(f"Skipping pruning: no active edges below threshold")

        if should_prune:
            print(f"\nBatch {i_batch} Pruning:")
            if args.optimized_spatial:
                print("Spatial Before:")
                print(graph.spatial_masks.reshape((N, N)))
                spat, _ = graph.update_masks_with_zero_extension(args.pruning_rate, similarity_matrix=spatial_edge_score, mask_type='spatial')
                print("Spatial After:")
                print(graph.spatial_masks.reshape((N, N)))
            else:
                spat = graph.spatial_masks
            
            old_N = graph.num_nodes
            old_node_order = graph.node_order.copy()
            
            # Store edge scores before node removal for normalization adjustment
            old_edge_score_before_removal, _ = spatial_edge_manager.get_edge_score_with_debug()
            old_beta = spatial_edge_manager.beta
            
            removed_ids = graph.check_and_remove_isolated_nodes(args.min_agents)
            if removed_ids:
                print(f"Graph structure changed! Rebuilding optimizer...")
                for removed_id in removed_ids:
                    role = node_id_to_role.get(removed_id, "Unknown Role")
                    print(f"    Removing isolated node: {removed_id} -- {role}")
                
                for removed_id in removed_ids:
                    if removed_id in node_id_to_role:
                        del node_id_to_role[removed_id]

                optimizer = torch.optim.Adam([graph.spatial_logits], lr=args.lr)
                new_N = graph.num_nodes
                if new_N != old_N:
                    old_id_to_index = {node_id: i for i, node_id in enumerate(old_node_order)}
                    removed_indices = [old_id_to_index[node_id] for node_id in removed_ids if node_id in old_id_to_index]
                    remaining_indices = [i for i in range(old_N) if i not in removed_indices]
                    print(f"Updating spatial_edge_manager matrix size from {old_N}x{old_N} to {new_N}x{new_N}")
                    spatial_edge_manager.remove_nodes_and_update_matrix(removed_indices, remaining_indices)
                    
                    # Calculate normalization adjustment for beta weight
                    # Get edge scores after node removal with current beta
                    new_edge_score_after_removal, debug_info = spatial_edge_manager.get_edge_score_with_debug()
                    
                    # Extract remaining values from old matrix to compare with new normalized values
                    remaining_mask = np.zeros(old_N, dtype=bool)
                    remaining_mask[remaining_indices] = True
                    remaining_old_values = old_edge_score_before_removal[np.ix_(remaining_indices, remaining_indices)]
                    
                    # Calculate the ratio of old vs new normalization ranges
                    old_range = remaining_old_values.max() - remaining_old_values.min()
                    new_range = new_edge_score_after_removal.max() - new_edge_score_after_removal.min()
                    
                    if old_range > 1e-8 and new_range > 1e-8:
                        # Calculate how the normalization has changed the scale
                        # We want to adjust beta to maintain the relative position of values w.r.t. threshold
                        scale_ratio = old_range / new_range
                        
                        # Adjust beta weight to compensate for normalization change
                        adjusted_beta = old_beta * scale_ratio
                        spatial_edge_manager.update_beta_weight(adjusted_beta)
                        
                        print(f"Normalization adjustment after node removal:")
                        print(f"  Old range: {old_range:.4f}, New range: {new_range:.4f}")
                        print(f"  Scale ratio: {scale_ratio:.4f}")
                        print(f"  Beta weight adjusted: {old_beta:.4f} -> {adjusted_beta:.4f}")
                        
                        # Recompute edge scores with adjusted beta
                        spatial_edge_score, _ = spatial_edge_manager.get_edge_score_with_debug()
                        print("Updated spatial edge_score matrix after beta adjustment:")
                        print(np.round(spatial_edge_score, 4))
            else:
                print("No isolated nodes found")
            pruning_count += 1

        ckpt = {'batch_index': i_batch + 1, 'spatial_masks': spat.tolist(), 'pruning_count': pruning_count}
        with open(ckpt_file, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2)
            
        log_entry = {'batch': i_batch, 'cost': Cost.instance().value, 'prompt_tokens': PromptTokens.instance().value,
                     'completion_tokens': CompletionTokens.instance().value, 'accuracy': accuracy,
                     'loss': torch.mean(torch.stack(losses)).item() if losses else 0.0, 'pruning_count': pruning_count}
        log_data.append(log_entry)
        save_log(log_file, log_data)

        if args.optimized_spatial:
            total_loss = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Batch time {time.time() - start_ts:.2f}s | Accuracy {accuracy:.4f} | Loss {log_entry['loss']:.4f}")

    print("All batches done.")

def get_kwargs(mode: str, N: int) -> dict:
    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        layer_num = max(2, int(N**0.5))
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj_matrix[i][j] = 1
        fixed_spatial_masks = adj_matrix
    elif mode == 'Star':
        matrix = [[0] * N for _ in range(N)]
        for i in range(1, N):
            matrix[0][i] = 1
        fixed_spatial_masks = matrix
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {"fixed_spatial_masks": fixed_spatial_masks}

if __name__ == '__main__':
    asyncio.run(main())

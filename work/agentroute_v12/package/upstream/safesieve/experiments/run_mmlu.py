import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List
import argparse
import random
import json
import time
import math
from pathlib import Path
import torch
import copy
import numpy as np

from SafeSieve.graph.graph import Graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from SafeSieve.utils.const import SafeSieve_ROOT
from SafeSieve.utils.globals import Time, Cost, PromptTokens, CompletionTokens, RoleModelTokens
from SafeSieve.utils.utils import nuclear_norm
from experiments.accuracy import Accuracy
from SafeSieve.graph.prompt_similarity import compute_prompt_similarity_matrix, compute_prompt_similarity_matrix_llm_sync
from SafeSieve.prompt.mmlu_prompt_set import MMLUPromptSet
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

def dataloader(dataset, batch_size: int, i_batch: int) -> List:
    start = i_batch * batch_size
    end = min(start + batch_size, len(dataset))
    return [dataset[i] for i in range(start, end)]


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1,
                        help="noise level")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--llm_name', type=str, default="deepseek-ai/DeepSeek-V3",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    
    # New pruning parameters from run_mmlu_1.py
    parser.add_argument('--min_agents', type=int, default=2, help='Minimum number of agents to keep')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use existing checkpoint file if available')
    parser.add_argument('--use_cost_log', action='store_true', help='Use existing cost_log file if available')
    parser.add_argument('--max_data', type=int, default=0, help='Maximum number of data points to use')
    
    # Pruning Parameters
    parser.add_argument('--start_pruning_batch', type=int, default=10, help='Starting batch for pruning')
    parser.add_argument('--max_pruning_times', type=int, default=7, help='Maximum number of pruning operations')
    parser.add_argument('--min_threshold', type=float, default=0.65, help='Minimum dynamic edge score threshold')
    parser.add_argument('--max_threshold', type=float, default=0.9, help='Maximum dynamic edge score threshold')
    parser.add_argument('--alpha_weight', type=float, default=0.5, help='Weight for similarity matrix in edge score')
    parser.add_argument('--beta_growth_rate', type=float, default=1, help='Growth rate for dynamic beta weight')
    
    # LLM similarity calculation parameters
    parser.add_argument('--use_llm_similarity', action='store_true', help='Use LLM expert scoring for similarity calculation')
    parser.add_argument('--similarity_model', type=str, default='deepseek-ai/DeepSeek-V3', help='LLM model for similarity calculation')
    parser.add_argument('--similarity_max_score', type=float, default=10.0, help='Maximum score for LLM similarity evaluation')
    
    # Parameters for heterogeneous LLM experiments
    parser.add_argument('--heterogeneous_llms', action='store_true', help='Enable heterogeneous LLM experiment')
    parser.add_argument('--llm_config', nargs='+', type=str, default=["deepseek-ai/DeepSeek-V3","gpt-4o-mini","LoRA/Qwen/Qwen2.5-72B-Instruct","@cf/meta/llama-3.1-8b-instruct","moonshotai/Kimi-K2-Instruct"], help='List of LLM models for heterogeneous experiment')
    
    args = parser.parse_args()
    result_path = SafeSieve_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args


async def safe_arun_with_retry(graph: Graph, input_dict: dict, num_rounds: int,
                                max_retries: int = 5, delay_seconds: int = 30):
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
    # Reset global metrics for independent run
    Cost.instance().value = 0
    PromptTokens.instance().value = 0
    CompletionTokens.instance().value = 0

    args = parse_args()
    RoleModelTokens.instance().value = {name:0 for name in args.llm_config}
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    limit_questions = 153
    
    print(f">>> Starting MMLU experiment, mode={mode}, llm={args.llm_name}")

    # Download and load dataset
    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    
    # Limit dataset size if specified
    if args.max_data > 0:
        dataset_train._total_df = dataset_train._total_df.head(args.max_data)
        dataset_val._total_df = dataset_val._total_df.head(args.max_data)
    
    print(f">>> Loaded train: {len(dataset_train)}, val: {len(dataset_val)} items, batch_size={args.batch_size}")

    TEST_MAX_DATA = 200  
    if 'TEST_MAX_DATA' in locals():
        dataset_train._total_df = dataset_train._total_df.head(TEST_MAX_DATA)
        dataset_val._total_df = dataset_val._total_df.head(TEST_MAX_DATA)
        print(f">>> [DEBUG] DATASET = {TEST_MAX_DATA} samples")

    # Prepare directories
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{SafeSieve_ROOT}/result/{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
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

    # Get MMLU-specific roles and compute similarity matrix
    try:
        # Try to use MMLU-specific prompt set
        roles = list(MMLUPromptSet.ROLE_DESCRIPTION.keys())
        roles = roles[:len(agent_names)]
        role_prompts = [MMLUPromptSet.ROLE_DESCRIPTION[role] for role in roles]
        print(f"MMLU roles: {roles}")
    except (ImportError, AttributeError) as e:
        # Fallback to generic roles if MMLU prompt set doesn't exist
        print(f"Warning: MMLUPromptSet not found ({e}), using generic roles")
        roles = [f"AnalyzeAgent_{i}" for i in range(len(agent_names))]
        role_prompts = [f"You are an expert analyzer specializing in multiple choice questions. Role {i}." for i in range(len(agent_names))]
    
    # Compute prompt similarity matrix
    if args.use_llm_similarity:
        print(f">>> Using LLM expert scoring for similarity calculation (model: {args.similarity_model})")
        similarity_matrix = compute_prompt_similarity_matrix_llm_sync(
            role_prompts, 
            task_type="MMLU",
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
    
    # Parameters for heterogeneous LLM experiments
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
    kwargs = get_kwargs(mode, N, roles)
    
    # Initialize spatial logits
    spatial_logits = torch.full((N*N,), 0.5, dtype=torch.float32)
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  heterogeneous_llms=args.heterogeneous_llms,
                  llm_config=args.llm_config if args.heterogeneous_llms else None,
                  **kwargs)
    
    # Log the initial agent-role mapping
    node_id_to_role = {node_id: role for node_id, role in zip(graph.node_order, roles)}
    print(">>> Agent to Role Mapping:")
    for idx, (node_id, role) in enumerate(zip(graph.node_order, roles)):
        llm = args.llm_config[idx % len(args.llm_config)] if args.heterogeneous_llms else args.llm_name
        print(f"    {node_id} -- {role} (LLM: {llm})")

    if args.optimized_spatial or args.optimized_temporal:
        # Use the enhanced training logic from run_mmlu_1.py
        graph.spatial_logits = torch.nn.Parameter(spatial_logits, requires_grad=True)
        optimizer = torch.optim.Adam([graph.spatial_logits], lr=args.lr)
        
        # Initialize EdgeMatrixManager
        edge_manager = EdgeMatrixManager(N, similarity_matrix, alpha=args.alpha_weight, beta=1.0, epsilon=1e-6)

        # Enhanced training with the new pruning algorithm
        await enhanced_train(graph=graph, dataset=dataset_train, args=args, kwargs=kwargs, edge_manager=edge_manager, similarity_matrix=similarity_matrix, node_id_to_role=node_id_to_role)
        
    print("Final spatial logits: ",graph.spatial_logits)
    print("Final temporal logits: ",graph.temporal_logits)
    print("Final spatial masks: ",graph.spatial_masks)
    print("Final temporal masks: ",graph.temporal_masks)
    print("Final spatial sparsity:",graph.spatial_masks.sum()/graph.spatial_masks.numel())
    print("Final temporal sparsity:",graph.temporal_masks.sum()/graph.temporal_masks.numel())
    
    score = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,limit_questions=limit_questions,eval_batch_size=args.batch_size)
    print(f"Score: {score}")


async def enhanced_train(graph: Graph, dataset: MMLUDataset, args, kwargs, edge_manager: EdgeMatrixManager, similarity_matrix, node_id_to_role: dict):
    """Enhanced training with the new pruning algorithm from run_mmlu_1.py"""
    
    # Restore checkpoint if exists
    start_batch = 0
    pruning_count = 0
    
    # Determine number of batches
    num_batches = math.ceil(len(dataset) / args.batch_size)

    total_correct = 0
    total_executed = 0
    accuracy_tracker = Accuracy()

    for i_batch in range(start_batch, num_batches):
        print(f"Batch {i_batch}", '-'*80)
        start_ts = time.time()
        tasks, correct_answers, regs = [], [], []

        batch = dataloader(dataset, args.batch_size, i_batch)
        if not batch:
            break

        N = graph.num_nodes if hasattr(graph, 'num_nodes') else len(graph.node_order)
        # Ensure edge_manager N matches current graph N
        if edge_manager.N != N:
            print(f"Updating edge_manager N from {edge_manager.N} to {N}")
            edge_manager.update_matrix_size(N)
        
        S_fixed = None
        if 'fixed_spatial_masks' in kwargs and kwargs['fixed_spatial_masks'] is not None:
            S_fixed = torch.tensor(kwargs['fixed_spatial_masks'], dtype=torch.float32)
            if S_fixed.numel() != N*N:
                S_fixed = S_fixed.reshape(-1)[:N*N].reshape((N, N))
            else:
                S_fixed = S_fixed.reshape((N, N))

        realized_list = []
        for record in batch:
            N = graph.num_nodes if hasattr(graph, 'num_nodes') else len(graph.node_order)
            realized = copy.deepcopy(graph)
            realized_list.append(realized)
            
            # Ensure realized uses correct logits size
            realized.spatial_logits = graph.spatial_logits
            if realized.spatial_logits.numel() != N * N:
                print(f"Warning: spatial_logits size {realized.spatial_logits.numel()} doesn't match N*N={N*N}")
                if realized.spatial_logits.numel() > N * N:
                    realized.spatial_logits = realized.spatial_logits[:N*N]
                else:
                    new_logits = torch.zeros(N*N, requires_grad=True)
                    new_logits[:realized.spatial_logits.numel()] = realized.spatial_logits
                    realized.spatial_logits = new_logits
            
            S = realized.spatial_logits.reshape((N, N))
            loss_s = nuclear_norm(S)
            reg_loss = loss_s
            
            # Convert record to input format
            input_dict = dataset.record_to_input(record)
            correct_answer = dataset.record_to_target_answer(record)
            
            correct_answers.append(correct_answer)
            tasks.append(asyncio.create_task(safe_arun_with_retry(realized, input_dict, args.num_rounds)))
            regs.append(reg_loss)

        raw = await asyncio.gather(*tasks)
        raw_ans, log_probs = zip(*raw)
        losses, utils = [], []

        for record, resp, lp, reg, true_answer, solved_task in zip(batch, raw_ans, log_probs, regs, correct_answers, realized_list):
            # Process MMLU answer
            pred_answer = dataset.postprocess_answer(resp[0])
            
            # Handle invalid answers: if not A, B, C, D, randomly select one
            valid_options = ['A', 'B', 'C', 'D']
            if pred_answer.upper() not in valid_options:
                pred_answer = random.choice(valid_options)
                print(f"[WARNING] Invalid answer detected, randomly selected: {pred_answer}")
            
            # Check if answer is correct
            is_correct = (pred_answer.upper() == true_answer.upper())
            total_correct += is_correct
            total_executed += 1
            current_accuracy = total_correct / total_executed
            
            # Update accuracy tracker
            accuracy_tracker.update(pred_answer, true_answer)
            
            utils.append(is_correct)
            losses.append(-lp * is_correct + reg)

            # Update edge manager with successful connections
            if is_correct:
                adj = solved_task.spatial_adj_matrix
                edge_manager.update_complementarity(adj, solved=True)

            print(f"Question: {dataset.record_to_input(record)['task'][:100]}...")
            print(f"Correct Answer: {true_answer}, Prediction: {pred_answer}, Correct: {is_correct}")

        # Update complementarity with batch results
        adj = graph.spatial_adj_matrix
        batch_has_correct = any(utils)
        edge_manager.update_complementarity(adj, batch_has_correct)

        # Print current status
        print(f"\nBatch {i_batch} Current Mask Matrix:")
        print("Spatial mask matrix:")
        print(graph.spatial_masks.reshape((N, N)))
        print(f"Current Accuracy: {current_accuracy:.4f}")
        print(f"Cost: {Cost.instance().value} | PromptTokens: {PromptTokens.instance().value} | CompletionTokens: {CompletionTokens.instance().value} \nRoleModelTokens: {RoleModelTokens.instance().value}")

        # Update dynamic beta weight
        dynamic_beta = 1.0 + (i_batch * args.batch_size * args.beta_growth_rate) / 100.0
        edge_manager.update_beta_weight(dynamic_beta)
        print(f"Dynamic beta weight: {dynamic_beta:.2f}")
        
        # Print edge score matrix
        edge_score, edge_debug = edge_manager.get_edge_score_with_debug()
        print("Edge score matrix (for pruning):")
        print(np.round(edge_score, 4))

        # Calculate dynamic threshold
        current_threshold = dynamic_threshold(
            beta=dynamic_beta,
            base_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            growth_rate=args.beta_growth_rate
        )
        print(f"Dynamic edge score threshold for this batch: {current_threshold:.4f}")

        # Smart pruning logic
        should_prune = False
        if (i_batch + 1) % args.imp_per_iterations == 0 and args.optimized_spatial:
            if i_batch + 1 >= args.start_pruning_batch:
                if pruning_count < args.max_pruning_times:
                    current_mask_2d = graph.spatial_masks.reshape((N, N)).cpu().numpy()
                    active_non_diagonal_mask = (current_mask_2d > 0) & (~np.eye(N, dtype=bool))
                    
                    active_edge_scores = edge_score[active_non_diagonal_mask]
                    
                    below_threshold_count = 0
                    if active_edge_scores.size > 0:
                        below_threshold = active_edge_scores < current_threshold
                        below_threshold_count = np.sum(below_threshold)
                    
                    print(f"\nBatch {i_batch} Pruning Analysis:")
                    print(f"Edge score threshold: {current_threshold}")
                    print(f"Active non-diagonal edges below threshold: {below_threshold_count}/{active_edge_scores.size}")
                    print(f"Pruning count: {pruning_count}/{args.max_pruning_times}")
                    
                    if below_threshold_count > 0:
                        below_threshold_scores = active_edge_scores[active_edge_scores < current_threshold]
                        print(f"Scores of active edges below threshold: {np.round(below_threshold_scores, 4)}")
                        should_prune = True
                        print(f"Pruning condition met: {below_threshold_count} active edges below threshold")
                    else:
                        print(f"Skipping pruning: no active edges below threshold")
                else:
                    print(f"Skipping pruning: reached max pruning times ({args.max_pruning_times})")
            else:
                print(f"Skipping pruning: not yet reached start pruning batch ({args.start_pruning_batch})")

        # Execute pruning
        if should_prune:
            print(f"\nBatch {i_batch} Pruning:")
            print("Before:")
            print("Spatial mask matrix:")
            print(graph.spatial_masks.reshape((N, N)))
            
            spat, _ = graph.update_masks_with_zero_extension(args.pruning_rate, similarity_matrix=edge_score)
            pruning_count += 1
            
            print("\nAfter:")
            print("Spatial mask matrix:")
            print(graph.spatial_masks.reshape((N, N)))
            
            # Check and remove isolated nodes
            print(f"\nBatch {i_batch} Checking isolated nodes:")
            old_N = graph.num_nodes
            old_node_order = graph.node_order.copy()
            
            removed_ids = graph.check_and_remove_isolated_nodes(args.min_agents)
            if removed_ids:
                print(f"Graph structure changed! Rebuilding optimizer...")
                for removed_id in removed_ids:
                    role = node_id_to_role.get(removed_id, "Unknown Role")
                    print(f"    Removing isolated node: {removed_id} -- {role}")
                
                # Update the role mapping
                for removed_id in removed_ids:
                    if removed_id in node_id_to_role:
                        del node_id_to_role[removed_id]

                new_N = graph.num_nodes
                if new_N != old_N:
                    old_id_to_index = {node_id: i for i, node_id in enumerate(old_node_order)}
                    removed_indices = [old_id_to_index[node_id] for node_id in removed_ids if node_id in old_id_to_index]
                    remaining_indices = [i for i in range(old_N) if i not in removed_indices]
                    
                    edge_manager.remove_nodes_and_update_matrix(removed_indices, remaining_indices)
                    print(f"EdgeMatrixManager updated: {old_N} -> {new_N} nodes")
            else:
                print("No isolated nodes found")

        print(f"Batch {i_batch} completed in {time.time() - start_ts:.2f}s")

    # Final results
    final_accuracy = total_correct / total_executed if total_executed > 0 else 0
    print(f"\n" + "="*80)
    print(f"TRAINING RESULTS:")
    print(f"Total Questions: {total_executed}")
    print(f"Correct Answers: {total_correct}")
    print(f"Final Training Accuracy: {final_accuracy:.4f}")
    print("="*80)


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int, roles: List[str] = None):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    # Set node_kwargs with roles
    if roles is not None:
        node_kwargs = [{'role': role} for role in roles]
    else:
        node_kwargs = [{'role': None} for _ in range(N)]
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        if roles is None:
            node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())

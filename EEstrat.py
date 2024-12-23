import argparse
import os
import json
import torch
import torch.nn as nn
import logging
from typing import Dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import create_early_exit_resnet, load_model
from src.utils import (
    count_parameters,
    parse_json,
    discretize_output,
    find_matching_conditional_probs,
    compute_f,
    get_all_path_from_head,
    from_dict_to_full_path,
    find_matching_posterior_probs,
    find_matching_past_keys,
    get_head_from_p_k,
    load_json_file,
    parse_model_path,
)
from src.dataset import create_data_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_to_json(data, filename):
    """Save data to a JSON file with indentation."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def estimate_posterior_probabilities(conditional_probs: Dict, num_classes: int) -> torch.Tensor:  
    """
    Compute posterior probabilities p_k,l for each class k and head l.
    
    Args:
        conditional_probs: Pre-computed conditional probabilities b_l Dict[str, Dict[Tuple[Tuple[float, ...], int], Dict[float, int]]]
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Posterior probability matrix of shape (num_classes, num_exits)
    """
    num_exits = len(conditional_probs)  # Number of exits from conditional_probs keys
    
    # Initialize posterior matrix p_k_l of shape (num_classes, num_exits)
    p_k_l = {}
    
    # Initialize with uniform prior for first exit
    p_k_l[()] = {}
    for i in range(num_classes):
        p_k_l[()][i] = 1 / num_classes
    
    # Compute p_k,l for each exit using only conditional probabilities
    for l in tqdm(range(1, num_exits + 1)):
        exit_name = f'exit{l}' if l < num_exits else 'final'
        for previous_outputs in conditional_probs[exit_name].keys():
            prev_outputs, label = previous_outputs
            p_k_l[previous_outputs] = {}
            for current_output in conditional_probs[exit_name][previous_outputs].keys():
                if current_output not in p_k_l[previous_outputs]:
                    p_k_l[previous_outputs][current_output] = {}
                prob = conditional_probs[exit_name][previous_outputs][current_output] # This is P(X_i_l = ξ_l | X_i_1 = ξ_1, ..., X_i_l-1 = ξ_l-1, Y = k)
                matching_probs = find_matching_conditional_probs(conditional_probs, exit_name, prev_outputs, current_output, num_classes) # This contains all P(X_i_l = ξ_l | X_i_1 = ξ_1, ..., X_i_l-1 = ξ_l-1, Y = k') for k' in {1, ..., K}
                # Get p_k,l-1
                if l == 1:
                    p_prev = torch.ones(num_classes) / num_classes
                else:
                    past_current_key = prev_outputs[-1]
                    p_prev = torch.zeros(num_classes)
                    for i in range(num_classes):
                        past_past_key_i = tuple((prev_outputs[:-1], i))
                        if past_past_key_i in p_k_l and past_current_key in p_k_l[past_past_key_i]:
                            p_prev[i] = p_k_l[past_past_key_i][past_current_key]
                        else:
                            p_prev[i] = 0
                
                numerator = prob * p_prev[label]
                denominator = torch.sum(matching_probs * p_prev)
                
                # assert that p_k_l[previous_outputs][current_output] is not already set to a value
                assert p_k_l[previous_outputs][current_output] == {}
                p_k_l[previous_outputs][current_output] = (numerator / denominator).item()

    return p_k_l

def estimate_conditional_probabilities(model: nn.Module, dataloader: torch.utils.data.DataLoader, num_bins: int) -> Dict:
    """
    Estimate conditional probabilities P(Xi_ℓ = ξ_ℓ | Xi_1 = ξ_1, ..., Xi_ℓ-1 = ξ_ℓ-1, Y = k)
    """
    logger.info("Starting conditional probability estimation")
    
    # Initialize counters for each exit
    conditional_probs = {f'exit{i}': {} for i in range(1, 4)}
    conditional_probs['final'] = {}
    
    model.eval()
    model.to("cuda")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Get model outputs for all exits
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = model(inputs) # Dict[str, Tensor]
            
            # Process each sample in the batch
            batch_size = inputs.size(0)
            for i in range(batch_size):
                prev_outputs = []
                label = labels[i].item()
                
                # Process each exit for this sample
                for exit_name, exit_output in outputs.items():
                    # Get and discretize the output
                    single_output = exit_output[i].unsqueeze(0) # [1, K]
                    discretized_output = discretize_output(single_output, num_bins)[0]
                    
                    # Create keys for conditional probability
                    condition_key = (tuple(prev_outputs), label)
                    
                    # Initialize nested dictionaries if needed
                    if condition_key not in conditional_probs[exit_name]:
                        conditional_probs[exit_name][condition_key] = {}
                    
                    if discretized_output not in conditional_probs[exit_name][condition_key]:
                        conditional_probs[exit_name][condition_key][discretized_output] = 0
                    
                    # Update count
                    conditional_probs[exit_name][condition_key][discretized_output] += 1
                    
                    # Add this output to previous outputs for next exit
                    prev_outputs.append(discretized_output)

    logger.info("Normalizing probabilities")
    # Normalize to get conditional probabilities
    for exit_name in tqdm(conditional_probs):
        for condition_key in conditional_probs[exit_name]:
            prev_outputs, label = condition_key
            
            # Calculate total count for this specific condition (prev_outputs AND label)
            total_count = sum(conditional_probs[exit_name][condition_key].values())
            
            # Normalize probabilities
            if total_count > 0:  # Add check to prevent division by zero
                for output_val in conditional_probs[exit_name][condition_key]:
                    count = conditional_probs[exit_name][condition_key][output_val]
                    # Normalize by total count of samples with same previous outputs AND label
                    conditional_probs[exit_name][condition_key][output_val] = count / total_count

    logger.info("Conditional probability estimation completed")
    return conditional_probs

def convert_conditional_probs_to_json(conditional_probs):
    """Convert conditional probabilities to JSON-serializable format."""
    json_conditional_probs = {}
    for exit_name, exit_probs in conditional_probs.items():
        json_conditional_probs[exit_name] = {}
        for condition_key, outputs in exit_probs.items():
            prev_outputs_str = str(condition_key[0])
            label = condition_key[1]
            key_str = f"{prev_outputs_str}_{label}"

            json_conditional_probs[exit_name][key_str] = {
                str(output): prob for output, prob in outputs.items()
            }
    return json_conditional_probs

def load_conditional_probs_from_json(filename):
    """Load and convert conditional probabilities from JSON file."""
    with open(filename, "r") as f:
        json_conditional_probs = json.load(f)

    conditional_probs = {}
    for exit_name, exit_probs in json_conditional_probs.items():
        conditional_probs[exit_name] = {}
        for key_str, outputs in exit_probs.items():
            prev_outputs_str, label = key_str.rsplit("_", 1)
            prev_outputs = eval(prev_outputs_str)
            label = int(label)

            conditional_probs[exit_name][(prev_outputs, label)] = {
                eval(output_str): prob for output_str, prob in outputs.items()
            }
    return conditional_probs

def compute_conditional_probs(model, dataloader, num_bins, exp_dir):
    """
    Compute or load cached conditional probabilities
    Args:
        model: The neural network model
        dataloader: DataLoader for the dataset
        num_bins: Number of bins for discretization
        exp_dir: Directory to save/load the probabilities
    Returns:
        dict: The conditional probabilities
    """
    conditional_probs_filename = os.path.join(exp_dir, "conditional_probabilities.json")
    
    if os.path.exists(conditional_probs_filename):
        logger.info("Loading cached conditional probabilities...")
        conditional_probs = load_conditional_probs_from_json(conditional_probs_filename)
        logger.info("Loaded cached conditional probabilities successfully.")
        conditional_probs = parse_json(conditional_probs)
    else:
        logger.info("Computing conditional probabilities...")
        # Compute conditional probabilities on the dataset
        conditional_probs = estimate_conditional_probabilities(
            model, dataloader, num_bins=num_bins
        )

        # Convert and save to JSON
        json_conditional_probs = convert_conditional_probs_to_json(conditional_probs)
        save_to_json(json_conditional_probs, conditional_probs_filename)
        logger.info("Computed and cached conditional probabilities.")
    
    return conditional_probs

def save_args(args, exp_dir):
    """Save arguments to a JSON file in the experiment directory"""
    args_dict = vars(args)
    args_path = os.path.join(exp_dir, "args.json")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

def convert_posterior_probs_to_json(p_k_l):
    """Convert posterior probabilities to JSON-serializable format."""
    json_posterior_probs = {}
    for prev_outputs, output_probs in p_k_l.items():
        prev_outputs_str = str(prev_outputs)
        json_posterior_probs[prev_outputs_str] = {}

        for output, prob in output_probs.items():
            output_str = str(output)
            json_posterior_probs[prev_outputs_str][output_str] = prob
    return json_posterior_probs

def load_posterior_probs_from_json(filename):
    """Load and convert posterior probabilities from JSON file."""
    with open(filename, "r") as f:
        json_posterior_probs = json.load(f)

    posterior_probs = {}
    for prev_outputs_str, output_probs in json_posterior_probs.items():
        prev_outputs = eval(prev_outputs_str)
        posterior_probs[prev_outputs] = {}

        for output_str, prob in output_probs.items():
            output = eval(output_str)
            posterior_probs[prev_outputs][output] = prob

    return posterior_probs

def compute_posterior_probs(conditional_probs, num_classes, exp_dir):
    """
    Compute or load cached posterior probabilities
    Args:
        conditional_probs: The conditional probabilities
        num_classes: Number of classes in the dataset
        exp_dir: Directory to save/load the probabilities
    Returns:
        dict: The posterior probabilities
    """
    posterior_probs_filename = os.path.join(exp_dir, "posterior_probabilities.json")
    
    if os.path.exists(posterior_probs_filename):
        logger.info("Loading cached posterior probabilities...")
        p_k_l = load_posterior_probs_from_json(posterior_probs_filename)
        logger.info("Loaded cached posterior probabilities successfully.")
    else:
        logger.info("Computing posterior probabilities...")
        # Compute p_k,l matrix using only conditional probabilities
        p_k_l = estimate_posterior_probabilities(
            conditional_probs=conditional_probs, num_classes=num_classes
        )

        # Save posterior probabilities
        json_posterior_probs = convert_posterior_probs_to_json(p_k_l)
        save_to_json(json_posterior_probs, posterior_probs_filename)
        logger.info("Computed and cached posterior probabilities.")
    
    return p_k_l

def get_confusion_matrix(model, dataloader, num_classes, head_number, model_path):
    """Generate confusion matrix for each head of the model on the dataloader
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader to evaluate the model on
        head_number (int): Number of the head to evaluate
        model_path (str): Path to the model file
    """
    # Create cache filename based on model path
    cache_path = os.path.splitext(model_path)[0] + "_confusion_matrix.pt"
    
    # Try to load cached confusion matrix
    if os.path.exists(cache_path):
        logger.info("Loading cached confusion matrix...")
        return torch.load(cache_path)

    logger.info("Computing confusion matrix...")
    confusion_matrix = torch.zeros((head_number, num_classes, num_classes))
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Computing confusion matrix", total=len(dataloader)):
            outputs = model(x)
            heads = ["exit1", "exit2", "exit3", "final"]
            for num, head in enumerate(heads):
                predicted_label = torch.argmax(outputs[head], dim=-1)
                confusion_matrix[num, y, predicted_label] += 1

    # Cache the computed confusion matrix
    torch.save(confusion_matrix, cache_path)
    logger.info(f"Saved confusion matrix to {cache_path}")
    
    return confusion_matrix

def compute_a_matrix(model, dataloader, num_classes, num_heads, a_matrix_type="confusion_matrix", lambda_factor=1.0, model_path=None, exp_dir=None):
    """Compute the A matrix (cost matrix) based on the specified type.
    
    Args:
        model (nn.Module): The early-exit model to evaluate
        dataloader (DataLoader): DataLoader containing the evaluation data
        num_classes (int): Number of classes in the dataset
        num_heads (int): Number of exits in the model
        a_matrix_type (str): Type of A matrix to compute. Options:
            - "confusion_matrix": Uses confusion matrix with zeros on diagonal
            - "ones_diagonal_zero": Matrix of ones with zeros on diagonal
        lambda_factor (float): Scaling factor to multiply the A matrix by. Default: 1.0
        model_path (str): Path to the model file for caching confusion matrix
        exp_dir (str): Directory to save cached matrices
    
    Returns:
        torch.Tensor: The computed A matrix scaled by lambda_factor
    
    Raises:
        ValueError: If an unknown a_matrix_type is provided
    """
    # Check number of heads if model is EarlyExitResNet
    if hasattr(model, 'aux_head1'):
        expected_heads = len([attr for attr in dir(model) if attr.startswith('aux_head')]) + 1
        if num_heads != expected_heads:
            logger.warning(f"num_heads ({num_heads}) does not match number of auxiliary heads + 1 ({expected_heads})")

    # Create cache filename
    cache_filename = f"{a_matrix_type}_A.json"
    if exp_dir:
        cache_path = os.path.join(exp_dir, cache_filename)
        if os.path.exists(cache_path):
            logger.info(f"Loading cached A matrix from {cache_path}")
            return torch.tensor(load_json_file(cache_path))

    if a_matrix_type == "confusion_matrix":
        if model_path is None:
            raise ValueError("model_path must be provided when using confusion_matrix A matrix type")
        A = get_confusion_matrix(model, dataloader, num_classes, num_heads, model_path)
        for head_idx in range(A.shape[0]):
            A[head_idx].fill_diagonal_(0)
    elif a_matrix_type == "ones_diagonal_zero":
        # Create base matrix of ones with zeros on diagonal
        base_matrix = torch.ones((num_classes, num_classes))
        base_matrix.fill_diagonal_(0)
        
        # Repeat the matrix for each head
        A = base_matrix.unsqueeze(0).repeat(num_heads, 1, 1)
    else:
        raise ValueError(f"Unknown A matrix type: {a_matrix_type}")
    
    A = lambda_factor * A
    
    # Cache the computed A matrix
    if exp_dir:
        with open(cache_path, "w") as f:
            json.dump(A.tolist(), f)
        logger.info(f"Saved A matrix to {cache_path}")
    
    return A

def compute_exit_costs(model, fixed_cost=None):
    """Compute exit costs based on parameter counts or use a fixed cost.
    
    Args:
        model (nn.Module): The early-exit model
        fixed_cost (float, optional): If provided, all exits will use this fixed cost value
        
    Returns:
        list: List of exit costs, either computed from parameters or fixed
    """
    # Get cumulative parameter counts
    cumulative_params = count_parameters(model)
    
    if fixed_cost is not None:
        # Use fixed cost for all exits
        exit_costs = [fixed_cost for _ in range(len(cumulative_params))]
    else:
        # Convert to incremental costs
        exit_costs = []
        prev_params = 0
        for params in cumulative_params:
            exit_costs.append(
                (params - prev_params) / cumulative_params[-1]
            )  # Normalize by total params
            prev_params = params
            
    return exit_costs

def compute_and_save_L_M_V_L(p_k_l, conditional_probs, A, exit_costs, num_classes, exp_dir):
    """Compute L_M and V_L values and save them to the experiment directory.
    
    Args:
        p_k_l (dict): Posterior probabilities
        conditional_probs (dict): Conditional probabilities
        A (torch.Tensor): Cost matrix
        exit_costs (list): List of exit costs for each head
        num_classes (int): Number of classes
        exp_dir (str): Directory to save results
        
    Returns:
        tuple: (L_M, V_L) dictionaries containing the computed values
    """
    # Check if L_M and V_L files already exist
    L_M_filename = os.path.join(exp_dir, "L_M.json")
    V_L_filename = os.path.join(exp_dir, "V_L.json")
    
    if os.path.exists(L_M_filename) and os.path.exists(V_L_filename):
        logger.warning("L_M and V_L files already exist in %s. Loading existing files instead of recomputing.", exp_dir)
        try:
            with open(L_M_filename, "r") as f:
                L_M = {eval(k): v for k, v in json.load(f).items()}
            with open(V_L_filename, "r") as f:
                V_L = {eval(k): v for k, v in json.load(f).items()}
            return L_M, V_L
        except Exception as e:
            logger.error("Failed to load existing L_M and V_L files: %s. Will recompute.", str(e))
    
    logger.info("Computing L_M and V_L values...")
    
    L_M = {}
    V_L = {}
    num_heads = len(exit_costs)
    
    for head in tqdm(range(num_heads)[::-1], desc="Processing heads"):
        paths = get_all_path_from_head(head, p_k_l)
        paths = from_dict_to_full_path(paths)
        for path in tqdm(paths, desc=f"Processing head {head}"):
            matched_p_k_l = find_matching_posterior_probs(
                p_k_l, path[0][:-1], path[0][-1], num_classes
            )
            if len(A.shape) == 2:
                L_M[path[0]] = compute_f(matched_p_k_l, A)
            else:
                L_M[path[0]] = compute_f(matched_p_k_l, A[head])
                
            if head != num_heads - 1:
                x = 0
                future_paths = get_all_path_from_head(head + 1, p_k_l)
                future_paths = find_matching_past_keys(future_paths, path[0])
                future_paths = from_dict_to_full_path(future_paths)
                unique_future_paths = set([key[0] for key in future_paths])
                for f_path in unique_future_paths:
                    conditional_prob_f_path = find_matching_conditional_probs(
                        conditional_probs,
                        get_head_from_p_k([f_path[:-1]]),
                        f_path[:-1],
                        f_path[-1],
                        num_classes,
                    )
                    vector_mul = torch.matmul(conditional_prob_f_path, matched_p_k_l)
                    chosen_L_M = L_M[f_path]
                    x += vector_mul * chosen_L_M
                    
                V_L[path[0]] = exit_costs[head + 1] + x
                L_M[path[0]] = min(L_M[path[0]], V_L[path[0]])
    
    # Save L_M to JSON
    logger.info("Saving L_M to JSON...")
    json_L_M = {str(key): float(value) if torch.is_tensor(value) else value 
                for key, value in L_M.items()}
    with open(L_M_filename, "w") as f:
        json.dump(json_L_M, f, indent=4)
    logger.info(f"Saved L_M to {L_M_filename}")
    
    # Save V_L to JSON
    logger.info("Saving V_L to JSON...")
    json_V_L = {str(key): float(value) if torch.is_tensor(value) else value 
                for key, value in V_L.items()}
    with open(V_L_filename, "w") as f:
        json.dump(json_V_L, f, indent=4)
    logger.info(f"Saved V_L to {V_L_filename}")
    
    return L_M, V_L

def main(args):
    logger.info("Starting experiment: %s", args.exp_name)

    dataset, base_model = parse_model_path(args.model_path)
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    
    # Create experiment directory and save arguments
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_args(args, exp_dir)
    logger.info("Created experiment directory at: %s", exp_dir)
    
    # Load the model
    model = load_model(args.model_path, num_classes, base_model)
    
    # Compute exit costs
    exit_costs = compute_exit_costs(model, args.fixed_cost)
    logger.info("Exit costs: %s", exit_costs)
    
    # Load datasets
    train_loader, test_loader = create_data_loaders(
        dataset_name=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42
    )
    
    # Compute conditional probabilities
    logger.info("Computing conditional probabilities")
    conditional_probs = compute_conditional_probs(
        model, train_loader, args.num_bins, exp_dir
    )
    
    # Compute posterior probabilities
    logger.info("Computing posterior probabilities")
    posterior_probs = compute_posterior_probs(
        conditional_probs, num_classes, exp_dir
    )

    # Compute A matrix
    logger.info("Computing A matrix")
    num_heads = len([attr for attr in dir(model) if attr.startswith('aux_head')]) + 1
    A = compute_a_matrix(
        model=model,
        dataloader=test_loader,
        num_classes=num_classes,
        num_heads=num_heads,
        a_matrix_type=args.a_matrix_type,
        lambda_factor=args.lambda_factor,
        model_path=args.model_path,
        exp_dir=exp_dir
    )
    
    # Compute L_M and V_L
    L_M, V_L = compute_and_save_L_M_V_L(
        p_k_l=posterior_probs,
        conditional_probs=conditional_probs,
        A=A,
        exit_costs=exit_costs,
        num_classes=num_classes,
        exp_dir=exp_dir
    )
    
    # Log configuration details
    logger.info("Configuration:")
    logger.info("- Model path: %s", args.model_path)
    logger.info("- Dataset: %s", dataset)
    logger.info("- Base model: %s", base_model)
    logger.info("- Number of classes: %d", num_classes)
    logger.info("- Number of bins: %d", args.num_bins)
    logger.info("- Batch size: %d", args.batch_size)
    logger.info("- A matrix type: %s", args.a_matrix_type)
    logger.info("- Lambda factor: %f", args.lambda_factor)
    logger.info("- Fixed cost: %s", args.fixed_cost)

    return conditional_probs, posterior_probs, A

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early exit strategy experiment")
    parser.add_argument('--model_path', type=str, default="checkpoints/best_model_cifar10_resnet18.pth",
                        help='Path to the model checkpoint')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--a_matrix_type', type=str, default="confusion_matrix",
                        choices=["confusion_matrix", "ones_diagonal_zero"],
                        help='Type of A matrix to compute')
    parser.add_argument('--lambda_factor', type=float, default=1.0,
                        help='Scaling factor for A matrix')
    parser.add_argument('--num_bins', type=int, default=11,
                        help='Number of bins')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for dataloaders')
    parser.add_argument('--fixed_cost', type=float, default=None,
                        help='Fixed cost to use for all exits. If None, costs are computed from parameters.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args)

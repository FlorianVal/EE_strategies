import re
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict, List, Tuple
import numpy as np
import ast
from collections.abc import Mapping
from tqdm import tqdm
import json
import logging
import os

def count_parameters(model: nn.Module) -> List[int]:
    """
    Count the cumulative number of parameters up to and including each exit in the model.
    
    Args:
        model (nn.Module): The early exit model.
    
    Returns:
        List[int]: Cumulative parameter counts for each exit.
    """
    exit_params = []
    total_params = 0
    
    # Track which layers belong to which exits
    exit_groups = {
        'exit1': ['conv1', 'bn1', 'layer1', 'aux_head1'],
        'exit2': ['layer2', 'aux_head2'],
        'exit3': ['layer3', 'aux_head3'],
        'final': ['layer4', 'fc']
    }
    
    # Helper function to check if a name belongs to any layers in the group
    def belongs_to_layer(name: str, layer_names: List[str]) -> bool:
        return any(layer in name for layer in layer_names)
    
    # Count parameters for each exit point
    for exit_name, layers in exit_groups.items():
        for name, module in model.named_modules():
            # Only count parameters for modules that have weights
            if hasattr(module, 'weight') or hasattr(module, 'bias'):
                # Check if this module belongs to the current exit group or any previous groups
                current_and_previous = []
                for key in exit_groups:
                    current_and_previous.extend(exit_groups[key])
                    if key == exit_name:
                        break
                
                if belongs_to_layer(name, current_and_previous):
                    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    total_params += params
        
        exit_params.append(total_params)
    
    return exit_params

def count_flops(model: nn.Module, input_size: Tuple[int, int, int]) -> List[int]:
    """
    Estimate cumulative FLOPs up to and including each exit in the model.
    
    Args:
        model (nn.Module): The early exit model.
        input_size (Tuple[int, int, int]): Input size (C, H, W).
    
    Returns:
        List[int]: Cumulative FLOPs counts for each exit.
    """
    def hook_fn(module, input, output):
        flops = 0
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out
            flops = (2 * module.in_channels * module.out_channels * 
                    module.kernel_size[0] * module.kernel_size[1] - 1) * output.size(2) * output.size(3)
        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features - out_features
            flops = (2 * module.in_features - 1) * module.out_features
        module.flops = flops
        module.output_size = output.size()

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Run a forward pass
    dummy_input = torch.randn(1, *input_size)
    _ = model(dummy_input)

    # Define layer groups for each exit
    exit_groups = {
        'exit1': ['conv1', 'bn1', 'layer1', 'aux_head1'],
        'exit2': ['layer2', 'aux_head2'],
        'exit3': ['layer3', 'aux_head3'],
        'final': ['layer4', 'fc']
    }

    def belongs_to_layer(name: str, layer_names: List[str]) -> bool:
        return any(layer in name for layer in layer_names)

    # Calculate cumulative FLOPs for each exit
    exit_flops = []
    total_flops = 0

    for exit_name, layers in exit_groups.items():
        for name, module in model.named_modules():
            if hasattr(module, 'flops'):
                # Check if this module belongs to the current exit group or any previous groups
                current_and_previous = []
                for key in exit_groups:
                    current_and_previous.extend(exit_groups[key])
                    if key == exit_name:
                        break
                
                if belongs_to_layer(name, current_and_previous):
                    total_flops += module.flops
        
        exit_flops.append(total_flops)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return exit_flops

def discretize_output(output: torch.Tensor, num_bins: int) -> List[Tuple[int, int]]:
    """
    Discretize the output of a single exit for a batch of data.
    
    Args:
        output (torch.Tensor): The output tensor from an exit (batch_size, num_classes).
        num_bins (int): Number of discretization bins.
    
    Returns:
        List[Tuple[int, int]]: List of (discretized max value, argmax) for each sample in the batch.
    """
    max_vals, argmaxes = output.max(dim=1)
    result = []
    for max_val, argmax in zip(max_vals.tolist(), argmaxes.tolist()):
        # Clamp between 0 and 1
        max_val = min(max(max_val, 0), 1)
        # Discretize to num_bins evenly spaced values
        discretized = round(max_val * (num_bins - 1)) / (num_bins - 1)
        # Round to 2 decimal places
        discretized = round(discretized * 100) / 100
        result.append((discretized, argmax))
    return tuple(result)

def estimate_conditional_probabilities(model: nn.Module, dataloader: torch.utils.data.DataLoader, num_bins: int) -> Dict:
    """
    Estimate conditional probabilities P(Xi_ℓ = ξ_ℓ | Xi_1 = ξ_1, ..., Xi_ℓ-1 = ξ_ℓ-1, Y = k)
    """
    print("\n=== Starting Conditional Probability Estimation ===")
    
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

    print("\n=== Normalizing Probabilities ===")
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

    print("\n=== Conditional Probability Estimation Complete ===")
    return conditional_probs

def compute_posterior_probabilities(conditional_probs: Dict, num_classes: int) -> torch.Tensor:
    """
    Compute posterior probabilities p_k,l for each class k and head l.
    
    Args:
        conditional_probs: Pre-computed conditional probabilities b_l Dict[str, Dict[Tuple[Tuple[float, ...], int], Dict[float, int]]]
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Posterior probability matrix of shape (num_classes, num_exits)
    """
    num_exits = len(conditional_probs)  # Number of exits from conditional_probs keys
    
    # Initialize posterior matrix p_k,l of shape (num_classes, num_exits)
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
                denominator = 0
                for key, matching_prob in matching_probs.items():
                    prev_out, matching_label, current_out = key
                    assert prev_out == prev_outputs
                    assert current_out == current_output
                    denominator += matching_prob * p_prev[matching_label]
                # assert that p_k_l[previous_outputs][current_output] is not already set to a value
                assert p_k_l[previous_outputs][current_output] == {}
                p_k_l[previous_outputs][current_output] = (numerator / denominator).item()

    return p_k_l

def  compute_f(p_k: torch.Tensor, A: torch.Tensor) -> float:
    """
    Compute f(p_k) = min_j [sum_i A_ij * p_i]
    
    Args:
        p_k: Probability vector over classes
        A: Cost matrix where A[i,j] is cost of predicting j when true class is i
    
    Returns:
        float: Minimum expected cost
    """
    costs = torch.matmul(p_k, A)  # Expected cost for each possible prediction
    return torch.min(costs).item()


def get_head_from_p_k(p_k: torch.Tensor, as_number: bool = False) -> int:
    """
    Get the head a p_k_l key
    Require a whole past key (past_key, label)
    """
    if p_k == ():
        depth = -1
    else:
        depth = len(p_k[0])
    if as_number:
        return depth
    if depth == 3:
        return "final"
    elif depth == 2:
        return "exit3"
    elif depth == 1:
        return "exit2"
    elif depth == 0:
        return "exit1"
    else:
        raise ValueError(f"Invalid depth: {depth}")

def compute_continuation_cost(p_k_l: Dict, costs: List[float], A: torch.Tensor, conditional_probs: Dict) -> Dict:
    """
    Compute continuation costs using dynamic programming equations (23)-(25) in backward order
    
    Ar 
    
    gs:
        p_k_l: Posterior probabilities for each exit
        costs: List of computational costs c_l for each exit 
        A: Cost matrix for classification errors
        conditional_probs: Pre-computed conditional probabilities b_l
    
    Returns:
        Dict: Continuation costs V_bar and L_bar for each exit and probability vector
    """
    num_exits = len(costs)
    
    # Initialize arrays for L_bar and V_bar
    L_bar = {}  # L̄_ℓ(p_ℓ) for each exit
    V_bar = {}  # V̄_ℓ(p_ℓ) for each non-final exit
    
    # Step 1: Initialize final exit (Eq. 25)
    print("\nInitializing final exit (L̄_M)")
    for prev_outputs in p_k_l.keys():
        if isinstance(prev_outputs, tuple) and len(prev_outputs) == num_exits - 1:
            L_bar[(prev_outputs, num_exits-1)] = compute_f(p_k_l[prev_outputs], A)
            print(f"L̄_M({prev_outputs}) = {L_bar[(prev_outputs, num_exits-1)]:.4f}")
    
    # Step 2: Work backwards through exits (Eqs. 23-24)
    print("\nWorking backwards through exits")
    for l in range(num_exits-2, -1, -1):  # Start from M-1 down to 0
        print(f"\nProcessing exit {l+1}")
        for prev_outputs in p_k_l.keys():
            if isinstance(prev_outputs, tuple) and len(prev_outputs) == l:
                # First compute V̄_ℓ(p_ℓ) according to Eq. 24
                v_bar = costs[l + 1]  # Start with c_ℓ+1
                
                # For each possible output at the next exit
                exit_name = f'exit{l+1}' if l < num_exits-2 else 'final'
                
                # Sum over all possible outputs and classes
                for condition_key in conditional_probs[exit_name].keys():
                    prev_out, label = condition_key
                    if prev_out == prev_outputs:
                        for xi_next, b_prob in conditional_probs[exit_name][condition_key].items():
                            # Get p_l+1 using stored values
                            next_outputs = prev_outputs + (xi_next,)
                            if next_outputs in p_k_l:
                                p_next = p_k_l[next_outputs]
                                # Use L_bar from previous computation (l+1)
                                L_next = L_bar.get((next_outputs, l+1), compute_f(p_next, A))
                                v_bar += b_prob * L_next
                
                V_bar[(prev_outputs, l)] = v_bar
                print(f"V̄_{l}({prev_outputs}) = {v_bar:.4f}")
                
                # Then compute L̄_ℓ(p_ℓ) = min[f(p_ℓ), V̄_ℓ(p_ℓ)] (Eq. 23)
                f_value = compute_f(p_k_l[prev_outputs], A)
                L_bar[(prev_outputs, l)] = min(f_value, v_bar)
                print(f"L̄_{l}({prev_outputs}) = min({f_value:.4f}, {v_bar:.4f}) = {L_bar[(prev_outputs, l)]:.4f}")
    
    return V_bar, L_bar

def select_optimal_head(p_k_l: Dict, costs: List[float], A: torch.Tensor, conditional_probs: Dict) -> int:
    """
    Select optimal exit head using dynamic programming.
    
    Args:
        p_k_l: Posterior probabilities for each exit
        costs: List of computational costs c_l for each exit
        A: Cost matrix for classification errors
        conditional_probs: Pre-computed conditional probabilities b_l
        
    Returns:
        int: Index of optimal exit head
    """
    # Compute continuation costs and optimal costs backwards
    V_bar, L_bar = compute_continuation_cost(p_k_l, costs, A, conditional_probs)
    
    # Follow minimum costs forward to select optimal head
    current_head = 0
    current_outputs = ()
    
    print("\nSelecting optimal head by following minimum costs forward:")
    while current_head < len(costs) - 1:
        f_value = compute_f(p_k_l[current_outputs], A)
        v_value = V_bar.get((current_outputs, current_head), float('inf'))
        
        print(f"\nAt exit {current_head + 1}:")
        print(f"  f(p_{current_head}) = {f_value:.4f} (cost of exiting now)")
        print(f"  V̄_{current_head}(p_{current_head}) = {v_value:.4f} (cost of continuing)")
        
        # According to Eq. 23, if f(p_l) <= V̄_l(p_l), we should exit now
        if f_value <= v_value:
            print(f"  -> Exit now at {current_head + 1}")
            return current_head
            
        print(f"  -> Continue to next exit")
        # Update for next iteration
        current_head += 1
        if current_outputs in p_k_l:
            next_output = max(p_k_l[current_outputs].items(), key=lambda x: x[1])[0]
            current_outputs = current_outputs + (next_output,)
    
    print(f"\nReached final exit {len(costs)}")
    return len(costs) - 1

def parse_json(data: Dict) -> Dict:
    new_dict = {}
    for key, value in data.items():
        # Convert the key from string to tuple/int
        try:
            new_key = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            new_key = key  # Keep the original key if it can't be converted

        # Recursively apply to nested dictionaries (or mappings)
        if isinstance(value, Mapping):  # This checks if value is dict-like
            new_dict[new_key] = parse_json(value)
        else:
            new_dict[new_key] = value
    return new_dict

def find_matching_conditional_probs(conditional_probs: Dict, exit_name: str, past_key: Tuple, current_key, num_classes: int) -> torch.Tensor:
    """
    Finds and returns a tensor of conditional probabilities that match the given past and current keys.
    """
    prob_tensor = torch.zeros(num_classes)
    assert not isinstance(list(conditional_probs[exit_name].keys())[0], str), f"The keys of the conditional_probs should be tuples : {list(conditional_probs.keys())}"
    for i in range(num_classes):
        if (past_key, i) in conditional_probs[exit_name] and current_key in conditional_probs[exit_name][(past_key, i)]:
            prob_tensor[i] = conditional_probs[exit_name][(past_key, i)][current_key]
    return prob_tensor

def find_matching_posterior_probs(posterior_probs: Dict, past_key: Tuple, current_key, num_classes: int) -> torch.Tensor:
    """
    Finds and returns a tensor of posterior probabilities that match the given past and current keys.
    Args:
        posterior_probs (Dict): A dictionary containing posterior probabilities.
        past_key (Tuple): The key representing the past state.
        current_key: The key representing the current state.
        num_classes (int): The number of classes.
    Returns:
        torch.Tensor: A tensor containing the matching posterior probabilities for each class.
    """
    prob_tensor = torch.zeros(num_classes)
    assert not isinstance(list(posterior_probs.keys())[0], str), "The keys of the posterior_probs should be tuples"
    for i in range(num_classes):
        hypothetical_key = (past_key, i)
        if hypothetical_key in posterior_probs and current_key in posterior_probs[hypothetical_key]:
            prob_tensor[i] = posterior_probs[hypothetical_key][current_key]
    return prob_tensor 

def find_matching_past_keys(probs_dict: Dict, past_key: Tuple) -> Dict:
    """Find all the past keys that match the given past key

    Args:
        probs_dict (Dict): The dictionary of probabilities of the form p_k_l
        past_key (Tuple): The past key to match

    Returns:
        Dict: A sub dict of probs_dict that contains all the past keys that correspond to the given past key
    """
    matched_entries = {}
    for key, value in probs_dict.items():
        if isinstance(key, str):
            key = eval(key)
        if key[0] == past_key:
            matched_entries[key] = value
    return matched_entries
"""
def find_V_L_cost(target_key: Tuple, V_L: Dict, num_classes: int = 10) -> List[float]:
    ""
    Find the cost of each class for a given key in V_L
    ""
    costs = torch.zeros(num_classes)
    for key, value in V_L.items():
        if isinstance(key, str):
            key = eval(key)
        if key[0] == target_key:
            costs[key[1]] = value
    return costs
"""
def get_all_path_from_head(head: int, posterior_probs:Dict) -> Dict:
    """Get all the paths that correspond to a specific head from posterior probs

    Args:
        head (int): The head to get the paths for
        posterior_probs (Dict): The posterior probabilities Dict

    Returns:
        Dict: A sub dict from posterior_probs that contains all the paths that correspond to the head
    """
    paths = {}
    for key, value in posterior_probs.items():
        if isinstance(key, str):
            key = eval(key)
        if get_head_from_p_k(key, as_number=True) == head:
            paths[key] = value
    return paths

def from_dict_to_full_path(path_dict:Dict) -> Dict:
    """Unwrap dict to get from "key":{"key":value} to "key_key":value

    Args:
        path_dict (Dict): The dictionary to unwrap (p_k_l)

    Returns:
        Dict: The unwrapped dictionary
    """
    full_path = {}
    for key, value in path_dict.items():
        if key == ():
            raise NotImplementedError("This function is not implemented for the first exit")
        if isinstance(key, str):
            key = eval(key)
        past_key, label = key
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_key, str):
                    sub_key = eval(sub_key)
                if past_key == ():  # If the past key is empty, we are at the first exit
                    full_path[((sub_key,), label)] = sub_value
                else:
                    full_path[((past_key + (sub_key,)), label)] = sub_value
    return full_path

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON content
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info("Successfully loaded %s", file_path)
        return data
    except Exception as e:
        logging.error("Failed to load %s: %s", file_path, str(e))
        raise

def parse_model_path(model_path: str) -> tuple:
    """Parse model path to extract dataset and base model.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        tuple: (dataset_name, base_model_name)
    """
    try:
        # get last directory name
        if os.path.isdir(model_path):
            dir_name = os.path.basename(model_path)
        else:
            dir_name = os.path.dirname(model_path)
        dataset = dir_name.split("_")[0]
        base_model = dir_name.split("_")[1]
        
        if dataset not in ["cifar10", "cifar100"]:
            raise ValueError(f"Dataset {dataset} not recognized. Expected 'cifar10' or 'cifar100'")
            
        if not base_model:
            raise ValueError("Model name not found in path")
            
        return dataset, base_model
    except Exception as e:
        logging.error("Failed to parse model path: %s", str(e))
        raise ValueError(f"Invalid model path format. Expected: path/to/model_datasetname_modelname.pth, got: {model_path}")

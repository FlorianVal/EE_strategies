import argparse
import os
import json
import torch
import logging
from tqdm import tqdm
from typing import Dict, List
from src.model import create_early_exit_resnet, load_model
from src.dataset import create_data_loaders
from src.utils import (
    compute_f,
    discretize_output,
    find_matching_posterior_probs,
    parse_json,
    load_json_file,
    parse_model_path,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_results(results: List[Dict], results_path: str):
    """Save inference results to a JSON file.
    
    Args:
        results (List[Dict]): List of inference results
        results_path (str): Path to save the results
    """
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Successfully saved results to %s", results_path)
    except Exception as e:
        logger.error("Failed to save results: %s", str(e))
        raise

def run_inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    p_k_l: Dict,
    v_l: Dict,
    A: torch.Tensor,
    num_bins: int,
    device: torch.device,
    num_classes: int
) -> List[Dict]:
    """Run inference on the test dataset with batch processing.
    
    Args:
        model (torch.nn.Module): The early exit model
        test_loader (DataLoader): Test data loader
        p_k_l (Dict): Posterior probabilities
        v_l (Dict): Continuation costs
        A (torch.Tensor): Cost matrix
        num_bins (int): Number of bins for discretization
        device (torch.device): Device to run inference on
        num_classes (int): Number of classes in the dataset
        
    Returns:
        List[Dict]: List of inference results for each sample
    """
    results = []
    model.to(device)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Running inference"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            outputs = model(inputs)

            # Process each sample in the batch
            for sample_idx in range(batch_size):
                past_key = ()
                data_result = {
                    "true_label": targets[sample_idx].item(),
                    "exits": []
                }

                for exit_name, output in outputs.items():
                    # Get output for current sample
                    sample_output = output[sample_idx:sample_idx+1]
                    discretized_output = discretize_output(sample_output, num_bins)
                    posterior_proba = find_matching_posterior_probs(
                        p_k_l, past_key, discretized_output, num_classes
                    )
                    past_key += (discretized_output,)
                    
                    if len(A.shape) == 2:
                        cost_of_stopping = compute_f(posterior_proba, A)
                    else:
                        exits = ["exit1", "exit2", "exit3", "final"]
                        cost_of_stopping = compute_f(posterior_proba, A[exits.index(exit_name)])
                    
                    exit_data = {
                        "exit_name": exit_name,
                        "discretized_output": discretized_output,
                        "cost_of_stopping": cost_of_stopping,
                    }
                    
                    if exit_name != "final":
                        try:
                            cost_of_continuation = v_l[past_key]
                        except KeyError:
                            cost_of_continuation = None
                        exit_data["cost_of_continuation"] = cost_of_continuation
                        exit_data["stopped"] = (
                            cost_of_continuation is not None
                            and cost_of_stopping < cost_of_continuation
                        )
                    else:
                        exit_data["stopped"] = True
                    
                    data_result["exits"].append(exit_data)
                
                results.append(data_result)
    
    return results

def main(args):
    logger.info("Starting inference for experiment: %s", args.exp_name)
    
    # Parse model path and set up parameters
    dataset, base_model = parse_model_path(args.model_path)
    num_classes = 10 if dataset == "cifar10" else 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load model
    model = load_model(args.model_path, num_classes, base_model)
    
    # Set up experiment directory
    exp_dir = os.path.join("experiments", args.exp_name)
    if not os.path.exists(exp_dir):
        logger.error("Experiment directory %s does not exist", exp_dir)
        raise FileNotFoundError(f"Experiment directory {exp_dir} not found")
    
    # Load necessary files
    p_k_l = parse_json(load_json_file(os.path.join(exp_dir, "posterior_probabilities.json")))
    v_l = parse_json(load_json_file(os.path.join(exp_dir, "V_L.json")))
    
    # Load A matrix based on type
    if args.a_matrix_type == "confusion_matrix":
        A_path = os.path.join(exp_dir, "confusion_matrix_A.json")
    else:
        A_path = os.path.join(exp_dir, f"{args.a_matrix_type}_A.json")
    A = torch.tensor(load_json_file(A_path))
    
    # Create data loader
    _, test_loader = create_data_loaders(
        dataset_name=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42
    )
    
    # Run inference
    logger.info("Running inference on test set")
    results = run_inference(
        model=model,
        test_loader=test_loader,
        p_k_l=p_k_l,
        v_l=v_l,
        A=A,
        num_bins=args.num_bins,
        device=device,
        num_classes=num_classes
    )
    
    # Save results
    results_path = os.path.join(exp_dir, "inference_results.json")
    save_results(results, results_path)
    
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early exit inference")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model_cifar10_resnet18.pth",
                      help="Path to the model checkpoint")
    parser.add_argument("--exp_name", type=str, required=True,
                      help="Name of the experiment")
    parser.add_argument("--a_matrix_type", type=str, default="confusion_matrix",
                      choices=["confusion_matrix", "ones_diagonal_zero"],
                      help="Type of A matrix to use")
    parser.add_argument("--num_bins", type=int, default=11,
                      help="Number of bins for discretization")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2,
                      help="Number of workers for data loading")
    
    args = parser.parse_args()
    main(args)

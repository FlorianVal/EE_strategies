import os
import itertools
import subprocess
import numpy as np
import argparse
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def create_lambda_experiment_configs() -> List[Dict[str, Any]]:
    """Create configurations for lambda variation experiment."""
    # Base configuration
    base_config = {
        "model_path": "checkpoints/best_model_cifar10_resnet18.pth",
        "a_matrix_type": "confusion_matrix",
        "num_bins": 11,
        "batch_size": 8,
        "fixed_cost": None,  # Use default cost computation
        "num_workers": 2,  # Added num_workers to base config
    }
    
    # Generate lambda values with fixed spacing between 0.001 and 0.1
    lambda_values = np.linspace(1, 2, 3)
    
    # Create configs for each lambda value
    configs = []
    for i, lambda_val in enumerate(lambda_values):
        config = base_config.copy()
        config["lambda_factor"] = float(lambda_val)
        config["exp_name"] = f"lambda_exp_{i}_lambda_{lambda_val:.3f}"
        configs.append(config)
    
    return configs

def create_numbins_experiment_configs() -> List[Dict[str, Any]]:
    """Create configurations for num_bins variation experiment."""
    # Base configuration
    base_config = {
        "model_path": "checkpoints/best_model_cifar10_resnet18.pth",
        "a_matrix_type": "ones_diagonal_zero",
        "lambda_factor": 1,  # Fixed lambda value
        "batch_size": 8,
        "fixed_cost": None,  # Use default cost computation
        "num_workers": 2,
    }
    
    # Generate num_bins values from 5 to 20
    num_bins_values = list(range(21, 40, 3))  # [5, 8, 11, 14, 17, 20]
    
    # Create configs for each num_bins value
    configs = []
    for i, num_bins in enumerate(num_bins_values):
        config = base_config.copy()
        config["num_bins"] = num_bins
        config["exp_name"] = f"numbins_exp_{i}_bins_{num_bins}"
        configs.append(config)
    
    return configs

def run_eestrat(config: Dict[str, Any]) -> None:
    """Run EEstrat.py with the given configuration."""
    cmd = ["python3", "EEstrat.py"]
    for key, value in config.items():
        if value is not None:  # Skip None values
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\nRunning EEstrat with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    subprocess.run(cmd, check=True)

def run_infer(config: Dict[str, Any]) -> None:
    """Run infer.py with the given configuration."""
    cmd = ["python3", "infer.py"]
    infer_config = {
        "model_path": config["model_path"],
        "exp_name": config["exp_name"],
        "a_matrix_type": config["a_matrix_type"],
        "num_bins": config["num_bins"],
        "batch_size": config["batch_size"],  # Use same batch size as EEstrat
        "num_workers": config["num_workers"]  # Use same num_workers as EEstrat
    }
    
    for key, value in infer_config.items():
        if value is not None:  # Skip None values
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\nRunning inference with config:")
    for key, value in infer_config.items():
        print(f"  {key}: {value}")
    subprocess.run(cmd, check=True)

def run_single_eestrat(config: Dict[str, Any]) -> None:
    """Run a single EEstrat experiment with error handling."""
    try:
        print(f"\nRunning EEstrat for experiment {config['exp_name']}")
        run_eestrat(config)
        print(f"Completed EEstrat for experiment {config['exp_name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running EEstrat for experiment {config['exp_name']}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in EEstrat for experiment {config['exp_name']}: {e}")
        return False

def run_single_infer(config: Dict[str, Any]) -> None:
    """Run a single inference experiment with error handling."""
    try:
        print(f"\nRunning inference for experiment {config['exp_name']}")
        run_infer(config)
        print(f"Completed inference for experiment {config['exp_name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for experiment {config['exp_name']}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in inference for experiment {config['exp_name']}: {e}")
        return False

def run_experiment_set(configs: List[Dict[str, Any]], experiment_name: str) -> None:
    """Run a set of experiments with the given configurations."""
    print(f"\nStarting experiment set: {experiment_name}")
    print(f"Number of configurations to run: {len(configs)}")
    
    # Create experiments directory if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    
    # Determine number of parallel processes (use half of available CPUs)
    num_processes = max(1, multiprocessing.cpu_count() // 2)
    print(f"Running with {num_processes} parallel processes")
    
    # First run all EEstrat configurations in parallel
    print("\nRunning all EEstrat configurations...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        eestrat_results = list(executor.map(run_single_eestrat, configs))
    
    # Filter configs based on successful EEstrat runs
    successful_configs = [config for config, success in zip(configs, eestrat_results) if success]
    print(f"\nSuccessfully completed {len(successful_configs)}/{len(configs)} EEstrat runs")
    
    if not successful_configs:
        print("No successful EEstrat runs, stopping experiment")
        return
    
    # Then run all inference configurations in parallel
    print("\nRunning all inference configurations...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        infer_results = list(executor.map(run_single_infer, successful_configs))
    
    # Report final results
    successful_infers = sum(infer_results)
    print(f"\nExperiment set completed:")
    print(f"- Total configurations: {len(configs)}")
    print(f"- Successful EEstrat runs: {len(successful_configs)}")
    print(f"- Successful inference runs: {successful_infers}")

def main():
    parser = argparse.ArgumentParser(description='Run EEStrategies experiments')
    parser.add_argument('--experiment', type=str, choices=['lambda', 'numbins'],
                      required=True, help='Type of experiment to run')
    args = parser.parse_args()
    
    if args.experiment == 'lambda':
        configs = create_lambda_experiment_configs()
        run_experiment_set(configs, "Lambda Variation Experiment")
    elif args.experiment == 'numbins':
        configs = create_numbins_experiment_configs()
        run_experiment_set(configs, "Num Bins Variation Experiment")

if __name__ == "__main__":
    main() 
import json
import matplotlib.pyplot as plt
import re
import os

# Directory containing experiment folders
experiments_dir = "experiments"
patterns = {
    "lambda_exp": r"lambda_exp_\d+_lambda_(\d+\.\d+)",
    "lambda_ones_exp": r"lambda_ones_exp_\d+_lambda_(\d+\.\d+)"
}

numbins_patterns = {
    "numbins_exp": r"numbins_exp_\d+_bins_(\d+)"
}

def get_experiment_configs(pattern):
    exp_configs = []
    for dirname in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, dirname)
        if os.path.isdir(exp_path):
            match = re.match(pattern, dirname)
            if match:
                param_val = float(match.group(1))
                results_file = os.path.join(exp_path, "inference_results.json")
                if os.path.exists(results_file):
                    exp_configs.append((results_file, param_val, dirname))
    return sorted(exp_configs, key=lambda x: x[1])

# Exit names and their computation costs
exit_names = ["exit1", "exit2", "exit3", "final"]
computation_costs = {"exit1": 1, "exit2": 2, "exit3": 3, "final": 4}

def compute_head_score(data):
    correct_counts = {exit_name: 0 for exit_name in exit_names}
    total_counts = 0
    # Calculate correct predictions per exit
    for sample in data:
        true_label = sample["true_label"]
        total_counts += 1
        for exit_data in sample["exits"]:
            exit_name = exit_data["exit_name"]
            predicted_label = exit_data["discretized_output"][1]
            if predicted_label == true_label:
                correct_counts[exit_name] += 1

    return {exit_name: correct_counts[exit_name] / total_counts for exit_name in exit_names}

def compute_stopped_score(data):
    correct_stopped = 0
    total_stopped = 0
    total_cost = 0
    
    for sample in data:
        true_label = sample["true_label"]
        for exit_data in sample["exits"]:
            exit_name = exit_data["exit_name"]
            if exit_data["stopped"]:
                predicted_label = exit_data["discretized_output"][1]
                if predicted_label == true_label:
                    correct_stopped += 1
                total_stopped += 1
                total_cost += computation_costs[exit_name]
                break

    stopped_accuracy = correct_stopped / total_stopped if total_stopped > 0 else 0
    average_cost = total_cost / total_stopped if total_stopped > 0 else 0
    return stopped_accuracy, average_cost

def process_experiments(pattern_name, pattern):
    exp_configs = get_experiment_configs(pattern)
    stopped_accuracies = []
    average_costs = []
    exp_names = []
    head_accuracies = None
    param_values = []

    for results_file, param_val, exp_name in exp_configs:
        with open(results_file, "r") as f:
            data = json.load(f)
        
        stopped_accuracy, average_cost = compute_stopped_score(data)
        stopped_accuracies.append(stopped_accuracy)
        average_costs.append(average_cost)
        exp_names.append(f"param={param_val}")
        param_values.append(param_val)
        
        if head_accuracies is None:
            head_accuracies = compute_head_score(data)

    return stopped_accuracies, average_costs, exp_names, head_accuracies, param_values

def plot_lambda_experiments():
    # Check if there are any lambda experiments
    has_experiments = False
    for pattern_name in patterns.keys():
        if any(re.match(patterns[pattern_name], dirname) for dirname in os.listdir(experiments_dir)):
            has_experiments = True
            break
    
    if not has_experiments:
        print("No lambda experiments found.")
        return

    plt.figure(figsize=(20, 8))
    colors = ['blue', 'red']
    markers = ['o', 's']

    # First subplot - Accuracy vs Budget
    plt.subplot(1, 2, 1)
    for idx, (pattern_name, pattern) in enumerate(patterns.items()):
        stopped_accuracies, average_costs, exp_names, head_accuracies, lambda_values = process_experiments(pattern_name, pattern)
        
        if stopped_accuracies:  # Only plot if we have data
            # Plot accuracy vs cost for each experiment as a line
            plt.plot(average_costs, stopped_accuracies, marker=markers[idx], color=colors[idx], 
                    label=f"{pattern_name.replace('_', ' ')}")
            
            # Plot head accuracies
            plt.plot([computation_costs[exit_name] for exit_name in exit_names],
                    [head_accuracies[exit_name] for exit_name in exit_names],
                    linestyle='--', color=colors[idx],
                    label=f"{pattern_name.replace('_', ' ')} Head")

    plt.xlabel("Mean number of layers (Budget)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Budget Comparison (Lambda Experiments)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot - Accuracy vs Lambda
    plt.subplot(1, 2, 2)
    for idx, (pattern_name, pattern) in enumerate(patterns.items()):
        stopped_accuracies, average_costs, exp_names, head_accuracies, lambda_values = process_experiments(pattern_name, pattern)
        
        if stopped_accuracies:  # Only plot if we have data
            plt.plot(lambda_values, stopped_accuracies, marker=markers[idx], color=colors[idx], 
                    label=f"{pattern_name.replace('_', ' ')}")

    plt.xlabel("Lambda Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Lambda Value")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/lambda_comparison_plots.png", bbox_inches='tight')
    plt.close()

def plot_numbins_experiments():
    # Check if there are any numbins experiments
    has_experiments = False
    for pattern_name in numbins_patterns.keys():
        if any(re.match(numbins_patterns[pattern_name], dirname) for dirname in os.listdir(experiments_dir)):
            has_experiments = True
            break
    
    if not has_experiments:
        print("No numbins experiments found.")
        return

    plt.figure(figsize=(20, 8))
    colors = ['green']
    markers = ['o']

    # First subplot - Accuracy vs Budget
    plt.subplot(1, 2, 1)
    for idx, (pattern_name, pattern) in enumerate(numbins_patterns.items()):
        stopped_accuracies, average_costs, exp_names, head_accuracies, numbins_values = process_experiments(pattern_name, pattern)
        
        if stopped_accuracies:  # Only plot if we have data
            # Plot accuracy vs cost for each experiment as a line
            plt.plot(average_costs, stopped_accuracies, marker=markers[idx], color=colors[idx], 
                    label=f"{pattern_name.replace('_', ' ')}")
            
            # Plot head accuracies
            plt.plot([computation_costs[exit_name] for exit_name in exit_names],
                    [head_accuracies[exit_name] for exit_name in exit_names],
                    linestyle='--', color=colors[idx],
                    label=f"{pattern_name.replace('_', ' ')} Head")

    plt.xlabel("Mean number of layers (Budget)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Budget Comparison (Num Bins Experiments)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot - Accuracy vs Num Bins
    plt.subplot(1, 2, 2)
    for idx, (pattern_name, pattern) in enumerate(numbins_patterns.items()):
        stopped_accuracies, average_costs, exp_names, head_accuracies, numbins_values = process_experiments(pattern_name, pattern)
        
        if stopped_accuracies:  # Only plot if we have data
            plt.plot(numbins_values, stopped_accuracies, marker=markers[idx], color=colors[idx], 
                    label=f"{pattern_name.replace('_', ' ')}")

    plt.xlabel("Number of Bins")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Bins")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiments/numbins_comparison_plots.png", bbox_inches='tight')
    plt.close()

def main():
    # Plot lambda experiments
    plot_lambda_experiments()
    
    # Plot numbins experiments
    plot_numbins_experiments()

if __name__ == "__main__":
    main()

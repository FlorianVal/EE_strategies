# Early Exit Strategy Testing

This repository implements and tests a method for early exit strategies in neural networks. Early exits allow a model to make predictions at intermediate layers, potentially saving computational resources when high confidence predictions can be made earlier in the network.

## Overview

The codebase consists of two main components:
1. `EEstrat.py`: Implements the calibration and computation of early exit strategies
2. `infer.py`: Handles inference using the calibrated early exit model

## Key Features

- **Early Exit Architecture**: Uses a modified ResNet architecture with multiple exit points
- **Probabilistic Decision Making**: 
  - Computes conditional probabilities for each exit point
  - Estimates posterior probabilities for decision making
  - Uses cost matrices to determine optimal exit points
- **Flexible Cost Configuration**:
  - Supports different types of cost matrices (confusion matrix or ones with zero diagonal)
  - Allows for fixed or parameter-based exit costs
  - Configurable lambda factor for cost scaling

## How It Works

1. **Calibration Phase** (`EEstrat.py`):
   - Takes a pre-trained model with multiple exit points
   - Computes conditional probabilities for each exit point using calibration data
   - Estimates posterior probabilities
   - Generates cost matrices (A matrix)
   - Computes continuation costs (V_L) and minimum expected costs (L_M)

2. **Inference Phase** (`infer.py`):
   - Uses the calibrated model and pre-computed probabilities
   - Makes exit decisions based on cost comparison
   - Determines whether to exit early or continue processing

## Usage

### Calibration
```bash
python EEstrat.py --model_path <path_to_model> \
                  --exp_name <experiment_name> \
                  --a_matrix_type <confusion_matrix|ones_diagonal_zero> \
                  --lambda_factor <factor> \
                  --num_bins <num_bins> \
                  --batch_size <batch_size>
```

### Inference
```bash
python infer.py --model_path <path_to_model> \
                --exp_name <experiment_name> \
                --a_matrix_type <matrix_type> \
                --num_bins <num_bins> \
                --batch_size <batch_size>
```

## Key Parameters

- `model_path`: Path to the pre-trained model checkpoint
- `exp_name`: Name of the experiment
- `a_matrix_type`: Type of cost matrix to use (confusion_matrix or ones_diagonal_zero)
- `lambda_factor`: Scaling factor for the cost matrix
- `num_bins`: Number of bins for discretizing outputs
- `batch_size`: Batch size for calibration/inference

## Output

The experiment results are saved in the `experiments/<exp_name>` directory, including:
- Conditional probabilities
- Posterior probabilities
- Cost matrices (A matrix)
- Inference results

## Calibration Outputs

During the calibration phase, several key components are computed and saved:
1. **Conditional Probabilities**: Probability of each exit's output given previous exits
2. **Posterior Probabilities**: Updated probabilities after observing exit outputs
3. **Cost Matrices**: Used to evaluate the cost of making decisions at each exit
4. **Continuation Costs (V_L)**: Cost of continuing to the next exit
5. **Minimum Expected Costs (L_M)**: Minimum expected cost for each state

## Supported Datasets

- CIFAR-10 (10 classes)
- CIFAR-100 (100 classes)

## Requirements

- PyTorch
- torchvision
- tqdm
- numpy 
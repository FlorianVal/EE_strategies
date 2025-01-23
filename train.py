import argparse
import logging
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import create_early_exit_resnet
from src.dataset import create_data_loaders
from src.utils import count_parameters, count_flops
from src.strategies import get_training_strategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str,
        strategy,
        scheduler=None,
        enable_logging=True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.strategy = strategy
        self.scheduler = scheduler
        self.enable_logging = enable_logging

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize TensorBoard writer if logging enabled
        logger.info(f"Logging enabled: {enable_logging}")
        self.writer = (
            SummaryWriter(os.path.join(save_dir, "tensorboard"))
            if enable_logging
            else None
        )

    def compute_entropy(self, logits):
        """Compute entropy of softmax probabilities."""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        # Set all BatchNorm layers to eval mode if batch size is 1
        if next(iter(self.train_loader))[0].size(0) == 1:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

        total_loss = 0.0
        correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        total = 0

        # For tracking entropy-based selection accuracy
        selected_correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        selected_total = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            if batch_idx == 0:
                logger.info(
                    f"Outputs for class {target[0].item()}: \n{[f'{x:.2f}' for x in outputs['exit1'][0].tolist()]}\n{[f'{x:.2f}' for x in outputs['exit2'][0].tolist()]}\n{[f'{x:.2f}' for x in outputs['exit3'][0].tolist()]}\n{[f'{x:.2f}' for x in outputs['final'][0].tolist()]}"
                )
            # Calculate loss using the strategy
            loss = self.strategy.compute_loss(outputs, target, self.criterion)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Step the scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()
                # Log learning rate
                if self.enable_logging and batch_idx % 100 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar(
                        "train/learning_rate",
                        current_lr,
                        epoch * len(self.train_loader) + batch_idx,
                    )

            total_loss += loss.item()
            total += target.size(0)

            # Calculate accuracy for each exit and entropy-based selection
            with torch.no_grad():
                # Calculate entropy for each exit
                entropies = {
                    exit_name: self.compute_entropy(exit_output)
                    for exit_name, exit_output in outputs.items()
                }

                # Find exit with minimum entropy for each sample
                entropy_tensor = torch.stack(
                    [entropy for entropy in entropies.values()]
                )
                min_entropy_indices = entropy_tensor.argmin(dim=0)

                for exit_name, exit_output in outputs.items():
                    pred = exit_output.argmax(dim=1)
                    correct[exit_name] += pred.eq(target).sum().item()

                    # Get index for current exit
                    exit_idx = list(outputs.keys()).index(exit_name)

                    # Find samples where this exit had minimum entropy
                    selected_mask = min_entropy_indices == exit_idx
                    if selected_mask.sum() > 0:
                        selected_correct[exit_name] += (
                            pred[selected_mask].eq(target[selected_mask]).sum().item()
                        )
                        selected_total[exit_name] += selected_mask.sum().item()

                    # Log number of samples selected at each exit
                    if self.enable_logging:
                        self.writer.add_scalar(
                            f"train/samples_selected_{exit_name}",
                            selected_mask.sum().item(),
                            epoch * len(self.train_loader) + batch_idx,
                        )

            # Log batch loss to TensorBoard
            if self.enable_logging:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Train Batch: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.6f}"
                )

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            exit_name: (100.0 * corr / total) for exit_name, corr in correct.items()
        }

        # Calculate entropy-based selection accuracies
        selected_accuracies = {
            exit_name: (100.0 * selected_correct[exit_name] / selected_total[exit_name])
            if selected_total[exit_name] > 0
            else 0.0
            for exit_name in selected_correct.keys()
        }

        # Log epoch metrics to TensorBoard
        if self.enable_logging:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            for exit_name, acc in accuracies.items():
                self.writer.add_scalar(f"train/accuracy_{exit_name}", acc, epoch)
            for exit_name, acc in selected_accuracies.items():
                self.writer.add_scalar(
                    f"train/selected_accuracy_{exit_name}", acc, epoch
                )
                self.writer.add_scalar(
                    f"train/epoch_samples_selected_{exit_name}",
                    selected_total[exit_name],
                    epoch,
                )

        return {
            "loss": avg_loss,
            "accuracies": accuracies,
            "selected_accuracies": selected_accuracies,
            "selected_samples": selected_total,
        }

    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        total = 0

        # For tracking entropy-based selection accuracy
        selected_correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        selected_total = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)

                # Calculate validation loss using the strategy
                loss = self.strategy.compute_loss(outputs, target, self.criterion)

                total_loss += loss.item()
                total += target.size(0)

                # Calculate entropy for each exit
                entropies = {
                    exit_name: self.compute_entropy(exit_output)
                    for exit_name, exit_output in outputs.items()
                }

                # Find exit with minimum entropy for each sample
                entropy_tensor = torch.stack(
                    [entropy for entropy in entropies.values()]
                )
                min_entropy_indices = entropy_tensor.argmin(dim=0)

                # Calculate accuracy for each exit
                for exit_name, exit_output in outputs.items():
                    pred = exit_output.argmax(dim=1)
                    correct[exit_name] += pred.eq(target).sum().item()

                    # Get index for current exit
                    exit_idx = list(outputs.keys()).index(exit_name)

                    # Find samples where this exit had minimum entropy
                    selected_mask = min_entropy_indices == exit_idx
                    if selected_mask.sum() > 0:
                        selected_correct[exit_name] += (
                            pred[selected_mask].eq(target[selected_mask]).sum().item()
                        )
                        selected_total[exit_name] += selected_mask.sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            exit_name: (100.0 * corr / total) for exit_name, corr in correct.items()
        }

        # Calculate entropy-based selection accuracies
        selected_accuracies = {
            exit_name: (100.0 * selected_correct[exit_name] / selected_total[exit_name])
            if selected_total[exit_name] > 0
            else 0.0
            for exit_name in selected_correct.keys()
        }

        # Log validation metrics to TensorBoard
        if self.enable_logging:
            self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)
            for exit_name, acc in accuracies.items():
                self.writer.add_scalar(f"val/accuracy_{exit_name}", acc, epoch)
            for exit_name, acc in selected_accuracies.items():
                self.writer.add_scalar(f"val/selected_accuracy_{exit_name}", acc, epoch)
                self.writer.add_scalar(
                    f"val/epoch_samples_selected_{exit_name}",
                    selected_total[exit_name],
                    epoch,
                )

        return {
            "loss": avg_loss,
            "accuracies": accuracies,
            "selected_accuracies": selected_accuracies,
            "selected_samples": selected_total,
        }

    def save_checkpoint(self, metrics: dict, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        path = os.path.join(self.save_dir, f"best_model.pt")
        torch.save(checkpoint, path)
        logger.info(f"Saved best model checkpoint to {path}")

    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        best_val_loss = float("inf")

        # Try to log model graph to TensorBoard, but don't fail if it doesn't work
        if self.enable_logging:
            try:
                dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
                self.writer.add_graph(self.model, dummy_input, use_strict_trace=False)
            except Exception as e:
                logger.warning(f"Failed to add model graph to TensorBoard: {e}")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.6f}")
            for exit_name, acc in train_metrics["accuracies"].items():
                logger.info(f"Train Accuracy ({exit_name}): {acc:.2f}%")
            for exit_name, acc in train_metrics["selected_accuracies"].items():
                samples = train_metrics["selected_samples"][exit_name]
                logger.info(
                    f"Train Selected Accuracy ({exit_name}): {acc:.2f}% ({samples} samples)"
                )

            # Validate
            val_metrics = self.validate(epoch)
            logger.info(f"Validation Loss: {val_metrics['loss']:.6f}")
            for exit_name, acc in val_metrics["accuracies"].items():
                logger.info(f"Validation Accuracy ({exit_name}): {acc:.2f}%")
            for exit_name, acc in val_metrics["selected_accuracies"].items():
                samples = val_metrics["selected_samples"][exit_name]
                logger.info(
                    f"Validation Selected Accuracy ({exit_name}): {acc:.2f}% ({samples} samples)"
                )

            # Save only if this is the best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(val_metrics, epoch)

        # Close TensorBoard writer
        if self.enable_logging:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Early Exit ResNet")

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "stl10", "stanford_cars"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    # Model parameters
    parser.add_argument(
        "--base-model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Base ResNet model",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer"
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--training-strategy",
        type=str,
        default="sum_ce",
        choices=[
            "sum_ce",
            "kl_consistency",
            "multi_truth_penalty",
            "adaptive_lambda",
            "lse_ce",
            "flattened_softmax",
            "sum_single_head_ce",
            "min_head_loss",
        ],
        help="Training strategy to use",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight for KL divergence penalty in kl_consistency strategy, or temperature parameter in lse_ce strategy",
    )
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=0.5,
        help="Weight for the uniformity loss in multi_truth_penalty strategy",
    )
    parser.add_argument(
        "--balance_weight",
        type=float,
        default=0.5,
        help="Weight for the head balance loss in multi_truth_penalty strategy",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Experiment name to be included in the save directory",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="ID of GPU to use for training (-1 for CPU)",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        default=False,
        help="Disable logging to tensorboard",
    )

    # Add new arguments for adaptive lambda strategy
    parser.add_argument(
        "--compare-with-final",
        action="store_true",
        help="Compare each head with final head instead of next head in adaptive lambda strategy",
    )
    parser.add_argument(
        "--lambda-increase-rate",
        type=float,
        default=1.1,
        help="Rate to increase lambda by when accuracy is worse",
    )
    parser.add_argument(
        "--lambda-decrease-rate",
        type=float,
        default=0.9,
        help="Rate to decrease lambda by when accuracy is better",
    )
    parser.add_argument(
        "--min-lambda",
        type=float,
        default=0.1,
        help="Minimum value for lambda in adaptive strategy",
    )
    parser.add_argument(
        "--max-lambda",
        type=float,
        default=10.0,
        help="Maximum value for lambda in adaptive strategy",
    )

    # Learning rate scheduler parameters
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cyclic"],
        help="Learning rate scheduler to use",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=0.1,
        help="Maximum learning rate for cyclic scheduler",
    )
    parser.add_argument(
        "--step-size-up",
        type=int,
        default=2000,
        help="Number of training iterations in the increasing half of a cycle",
    )
    parser.add_argument(
        "--cycle-mode",
        type=str,
        default="triangular",
        choices=["triangular", "triangular2", "exp_range"],
        help="Mode for cyclic learning rate",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device based on gpu-id
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create save directory with timestamp and experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name_str = f"_{args.exp_name}" if args.exp_name else ""
    save_dir = os.path.join(
        args.save_dir,
        f"{timestamp}_{args.dataset}_{args.base_model}_{exp_name_str}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save experiment arguments to JSON
    args_dict = vars(args)
    args_dict["device"] = str(device)  # Add device info
    with open(os.path.join(save_dir, "experiment_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
    logger.info(
        f"Saved experiment arguments to {os.path.join(save_dir, 'experiment_args.json')}"
    )

    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create model
    model = create_early_exit_resnet(
        num_classes=num_classes, base_model=args.base_model
    ).to(device)

    # Log model information
    param_counts = count_parameters(model)
    logger.info("Cumulative parameters at each exit:")
    for i, count in enumerate(param_counts, 1):
        exit_name = f"exit{i}" if i < len(param_counts) else "final"
        logger.info(f"{exit_name}: {count:,} parameters")

    # Get training strategy
    strategy_kwargs = {}
    if args.training_strategy == "kl_consistency":
        strategy_kwargs["beta"] = args.beta
    if args.training_strategy == "multi_truth_penalty":
        strategy_kwargs["uniformity_weight"] = args.uniformity_weight
        strategy_kwargs["balance_weight"] = args.balance_weight
    if args.training_strategy == "adaptive_lambda":
        strategy_kwargs.update(
            {
                "num_heads": len(param_counts),
                "compare_with_final": args.compare_with_final,
                "lambda_increase_rate": args.lambda_increase_rate,
                "lambda_decrease_rate": args.lambda_decrease_rate,
                "min_lambda": args.min_lambda,
                "max_lambda": args.max_lambda,
            }
        )
    if args.training_strategy == "lse_ce":
        strategy_kwargs["beta"] = args.beta
    strategy = get_training_strategy(args.training_strategy, **strategy_kwargs)

    # Create optimizer and criterion using the strategy
    optimizer = strategy.create_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = strategy.create_criterion()

    # Create scheduler if specified
    scheduler = None
    if args.scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr,
            max_lr=args.max_lr,
            step_size_up=args.step_size_up,
            mode=args.cycle_mode,
            cycle_momentum=True if args.momentum > 0 else False,
        )
        logger.info(
            f"Created cyclic learning rate scheduler with base_lr={args.lr}, "
            f"max_lr={args.max_lr}, step_size_up={args.step_size_up}, mode={args.cycle_mode}"
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        strategy=strategy,
        scheduler=scheduler,  # Pass scheduler to trainer
        enable_logging=not args.disable_logging,  # Pass enable_logging to trainer
    )

    # Train model
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()

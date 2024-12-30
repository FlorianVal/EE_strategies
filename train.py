import argparse
import logging
import os
import json
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import create_early_exit_resnet
from src.dataset import create_data_loaders
from src.utils import count_parameters, count_flops

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """Abstract base class for different training strategies."""

    @abstractmethod
    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """Create the optimizer for the training strategy."""
        pass

    @abstractmethod
    def create_criterion(self) -> nn.Module:
        """Create the loss criterion for the training strategy."""
        pass

    @abstractmethod
    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        """Compute the loss based on the strategy."""
        pass


class SumCEStrategy(TrainingStrategy):
    """Training strategy that uses sum of cross-entropy losses from all exits."""

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
  
    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=target.device)
        for exit_output in outputs.values():
            loss += criterion(exit_output, target)
        return loss


class KLConsistencyStrategy(TrainingStrategy):
    """Training strategy that uses CE loss + KL divergence penalty between consecutive exits."""

    def __init__(self, beta: float = 0.5):
        """Initialize with beta parameter for KL penalty weight."""
        self.beta = beta

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        # Initialize loss with CE for each exit
        loss = torch.tensor(0.0, device=target.device)

        # Sort exits by name to ensure consistent ordering
        exit_names = sorted(outputs.keys())

        # Add CE loss for each exit
        for exit_name in exit_names:
            loss += criterion(outputs[exit_name], target)

        # Add KL divergence penalty between consecutive exits
        for i in range(len(exit_names) - 1):
            current_exit = outputs[exit_names[i]]
            next_exit = outputs[exit_names[i + 1]]

            # Apply softmax to get probability distributions
            current_probs = torch.nn.functional.softmax(current_exit, dim=1)
            next_probs = torch.nn.functional.softmax(next_exit, dim=1)

            # Calculate KL divergence
            kl_div = torch.nn.functional.kl_div(
                current_probs.log(), next_probs, reduction="batchmean"
            )

            loss += self.beta * kl_div

        return loss


# Factory function to get training strategy
def get_training_strategy(strategy_name: str, **kwargs) -> TrainingStrategy:
    strategies = {
        "sum_ce": lambda: SumCEStrategy(),
        "kl_consistency": lambda: KLConsistencyStrategy(**kwargs),
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown training strategy: {strategy_name}")
    return strategies[strategy_name]()


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
        strategy: TrainingStrategy,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.strategy = strategy

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)

            # Calculate loss using the strategy
            loss = self.strategy.compute_loss(outputs, target, self.criterion)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total += target.size(0)

            # Calculate accuracy for each exit
            with torch.no_grad():
                for exit_name, exit_output in outputs.items():
                    pred = exit_output.argmax(dim=1)
                    correct[exit_name] += pred.eq(target).sum().item()

            # Log batch loss to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Train Batch: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.6f}"
                )

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            exit_name: (100.0 * corr / total)
            for exit_name, corr in correct.items()
        }

        # Log epoch metrics to TensorBoard
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        for exit_name, acc in accuracies.items():
            self.writer.add_scalar(f"train/accuracy_{exit_name}", acc, epoch)

        return {"loss": avg_loss, "accuracies": accuracies}

    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)

                # Calculate validation loss using the strategy
                loss = self.strategy.compute_loss(
                    outputs, target, self.criterion
                )

                total_loss += loss.item()
                total += target.size(0)

                # Calculate accuracy for each exit
                for exit_name, exit_output in outputs.items():
                    pred = exit_output.argmax(dim=1)
                    correct[exit_name] += pred.eq(target).sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            exit_name: (100.0 * corr / total)
            for exit_name, corr in correct.items()
        }

        # Log validation metrics to TensorBoard
        self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        for exit_name, acc in accuracies.items():
            self.writer.add_scalar(f"val/accuracy_{exit_name}", acc, epoch)

        return {"loss": avg_loss, "accuracies": accuracies}

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
        try:
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            self.writer.add_graph(
                self.model, dummy_input, use_strict_trace=False
            )
        except Exception as e:
            logger.warning(f"Failed to add model graph to TensorBoard: {e}")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.6f}")
            for exit_name, acc in train_metrics["accuracies"].items():
                logger.info(f"Train Accuracy ({exit_name}): {acc:.2f}%")

            # Validate
            val_metrics = self.validate(epoch)
            logger.info(f"Validation Loss: {val_metrics['loss']:.6f}")
            for exit_name, acc in val_metrics["accuracies"].items():
                logger.info(f"Validation Accuracy ({exit_name}): {acc:.2f}%")

            # Save only if this is the best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(val_metrics, epoch)

        # Close TensorBoard writer
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
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay"
    )
    parser.add_argument(
        "--training-strategy",
        type=str,
        default="sum_ce",
        choices=[
            "sum_ce",
            "kl_consistency",
        ],  # Add more strategies as they are implemented
        help="Training strategy to use",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight for KL divergence penalty in kl_consistency strategy",
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create save directory with timestamp and experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name_str = f"_{args.exp_name}" if args.exp_name else ""
    save_dir = os.path.join(
        args.save_dir,
        f"{args.dataset}_{args.base_model}{exp_name_str}_{timestamp}",
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
    strategy = get_training_strategy(args.training_strategy, **strategy_kwargs)

    # Create optimizer and criterion using the strategy
    optimizer = strategy.create_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = strategy.create_criterion()

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
    )

    # Train model
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()

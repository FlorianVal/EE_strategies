import argparse
import logging
import os
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import create_early_exit_resnet
from src.dataset import create_data_loaders
from src.utils import count_parameters, count_flops

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingStrategy(ABC):
    """Abstract base class for different training strategies."""
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create the optimizer for the training strategy."""
        pass
    
    @abstractmethod
    def create_criterion(self) -> nn.Module:
        """Create the loss criterion for the training strategy."""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: dict, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Compute the loss based on the strategy."""
        pass

class SumCEStrategy(TrainingStrategy):
    """Training strategy that uses sum of cross-entropy losses from all exits."""
    
    def create_optimizer(self, model: nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs: dict, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        loss = 0
        for exit_output in outputs.values():
            loss += criterion(exit_output, target)
        return loss

# Factory function to get training strategy
def get_training_strategy(strategy_name: str) -> TrainingStrategy:
    strategies = {
        'sum_ce': SumCEStrategy(),
        # Add more strategies here as they are implemented
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown training strategy: {strategy_name}")
    return strategies[strategy_name]

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
        strategy: TrainingStrategy
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
        
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
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
            
            if batch_idx % 100 == 0:
                logger.info(f'Train Batch: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.6f}')
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {exit_name: (100. * corr / total) for exit_name, corr in correct.items()}
        
        return {
            "loss": avg_loss,
            "accuracies": accuracies
        }
    
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = {"exit1": 0, "exit2": 0, "exit3": 0, "final": 0}
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                # Calculate validation loss using the strategy
                loss = self.strategy.compute_loss(outputs, target, self.criterion)
                
                total_loss += loss.item()
                total += target.size(0)
                
                # Calculate accuracy for each exit
                for exit_name, exit_output in outputs.items():
                    pred = exit_output.argmax(dim=1)
                    correct[exit_name] += pred.eq(target).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {exit_name: (100. * corr / total) for exit_name, corr in correct.items()}
        
        return {
            "loss": avg_loss,
            "accuracies": accuracies
        }
    
    def save_checkpoint(self, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = os.path.join(self.save_dir, f'best_model.pt')
        torch.save(checkpoint, path)
        logger.info(f"Saved best model checkpoint to {path}")
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.6f}")
            for exit_name, acc in train_metrics['accuracies'].items():
                logger.info(f"Train Accuracy ({exit_name}): {acc:.2f}%")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation Loss: {val_metrics['loss']:.6f}")
            for exit_name, acc in val_metrics['accuracies'].items():
                logger.info(f"Validation Accuracy ({exit_name}): {acc:.2f}%")
            
            # Save only if this is the best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(val_metrics)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Early Exit ResNet')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                      help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--base-model', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50'],
                      help='Base ResNet model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                      help='Weight decay')
    parser.add_argument('--training-strategy', type=str, default='sum_ce',
                      choices=['sum_ce'],  # Add more strategies as they are implemented
                      help='Training strategy to use')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Create model
    model = create_early_exit_resnet(
        num_classes=num_classes,
        base_model=args.base_model
    ).to(device)
    
    # Log model information
    param_counts = count_parameters(model)
    logger.info("Cumulative parameters at each exit:")
    for i, count in enumerate(param_counts, 1):
        exit_name = f"exit{i}" if i < len(param_counts) else "final"
        logger.info(f"{exit_name}: {count:,} parameters")
    
    # Get training strategy
    strategy = get_training_strategy(args.training_strategy)
    
    # Create optimizer and criterion using the strategy
    optimizer = strategy.create_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    criterion = strategy.create_criterion()
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.base_model}_{timestamp}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        strategy=strategy
    )
    
    # Train model
    trainer.train(args.epochs)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from abc import ABC, abstractmethod

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

            # Add small epsilon to prevent log(0)
            current_exit = current_exit.clamp(min=1e-7)
            next_exit = next_exit.clamp(min=1e-7)

            # Calculate KL divergence
            kl_div = torch.nn.functional.kl_div(
                current_exit.log(), next_exit, reduction="batchmean"
            )

            # Check for nan and replace with 0 if needed
            if torch.isnan(kl_div):
                print(f"Warning: NaN detected in KL div at exit {i}, setting to 0")
                kl_div = torch.tensor(0.0, device=target.device)

            loss += self.beta * kl_div

        return loss


class MultiTruthPenaltyStrategy(TrainingStrategy):
    """Training strategy using KL-based losses for head selection and uniformity."""

    def __init__(self, uniformity_weight: float = 0.5, balance_weight: float = 0.1):
        """Initialize with weights for the loss components."""
        self.uniformity_weight = uniformity_weight
        self.balance_weight = balance_weight
        self.eps = 1e-8  # Small epsilon to prevent numerical instability
        self.debug_logs = True  # Flag to control logging

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

    def compute_metric(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence metric (maximum probability) for each sample."""
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        if self.debug_logs:
            logger.info(f"Probabilities shape: {probs.shape}")
            logger.info(f"Max probabilities shape: {max_probs.shape}")
            logger.info(
                f"Max probability range: [{max_probs.min():.4f}, {max_probs.max():.4f}]"
            )
        return max_probs

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        if self.debug_logs:
            logger.info("\n=== Starting MultiTruthPenaltyStrategy Loss Computation ===")
            logger.info(f"Target shape: {target.shape}")

        # Stack outputs to [H, B, K] tensor
        exit_names = sorted(outputs.keys())
        stacked_outputs = torch.stack([outputs[name] for name in exit_names])
        num_heads, batch_size, num_classes = stacked_outputs.shape

        if self.debug_logs:
            logger.info(f"\n1. Initial Setup:")
            logger.info(f"Number of heads (H): {num_heads}")
            logger.info(f"Batch size (B): {batch_size}")
            logger.info(f"Number of classes (K): {num_classes}")
            logger.info(f"Stacked outputs shape: {stacked_outputs.shape}")
            logger.info(
                f"Logits range: [{stacked_outputs.min():.4f}, {stacked_outputs.max():.4f}]"
            )

        # 1. Compute confidence metrics and find maximum confidence heads
        metrics = torch.stack(
            [self.compute_metric(exit_output) for exit_output in stacked_outputs]
        )  # Shape: [H, B]
        max_head_indices = torch.argmax(metrics, dim=0)  # Shape: [B]

        if self.debug_logs:
            logger.info(f"\n2. Confidence Metrics:")
            logger.info(f"Metrics tensor shape: {metrics.shape}")
            logger.info(f"Metrics range: [{metrics.min():.4f}, {metrics.max():.4f}]")
            logger.info(f"Max head indices shape: {max_head_indices.shape}")
            logger.info(
                f"Head selection distribution: {torch.bincount(max_head_indices, minlength=num_heads).tolist()}"
            )

        # 2. Compute main classification loss
        batch_indices = torch.arange(batch_size, device=target.device)
        selected_logits = stacked_outputs[max_head_indices, batch_indices]
        main_loss = criterion(selected_logits, target)

        if self.debug_logs:
            logger.info(f"\n3. Main Classification Loss:")
            logger.info(f"Selected logits shape: {selected_logits.shape}")
            logger.info(f"Main loss value: {main_loss.item():.4f}")

        # 3. Compute uniformity loss for non-selected heads
        head_mask = torch.ones((num_heads, batch_size), device=target.device)
        head_mask[max_head_indices, batch_indices] = 0
        non_selected_mask = head_mask.bool()
        non_selected_probs = torch.softmax(stacked_outputs, dim=2)[non_selected_mask]

        if self.debug_logs:
            logger.info(f"\n4. Non-selected Heads Processing:")
            logger.info(f"Head mask shape: {head_mask.shape}")
            logger.info(f"Non-selected probabilities shape: {non_selected_probs.shape}")
            logger.info(
                f"Non-selected probs range: [{non_selected_probs.min():.4f}, {non_selected_probs.max():.4f}]"
            )

        # Compute KL divergence with uniform distribution
        uniform_dist = torch.ones(num_classes, device=target.device) / num_classes
        uniformity_loss = torch.mean(
            torch.sum(
                non_selected_probs
                * (
                    torch.log(non_selected_probs + self.eps)
                    - torch.log(uniform_dist + self.eps)
                ),
                dim=1,
            )
        )

        if self.debug_logs:
            logger.info(f"\n5. Uniformity Loss:")
            logger.info(f"Uniform distribution shape: {uniform_dist.shape}")
            logger.info(f"Uniformity loss value: {uniformity_loss.item():.4f}")

        # 4. Compute head balance loss
        head_counts = torch.bincount(max_head_indices, minlength=num_heads).to(
            target.device
        )
        head_probs = (head_counts.float() / batch_size) + self.eps
        head_probs = head_probs / head_probs.sum()  # Renormalize

        if self.debug_logs:
            logger.info(f"\n6. Head Balance:")
            logger.info(f"Head counts: {head_counts.tolist()}")
            logger.info(f"Head probabilities: {head_probs.tolist()}")

        uniform_head_dist = torch.ones_like(head_probs) / num_heads
        balance_loss = torch.sum(
            head_probs
            * (
                torch.log(head_probs + self.eps)
                - torch.log(uniform_head_dist + self.eps)
            )
        )
        if self.debug_logs:
            logger.info(f"\n7. Balance Loss:")
            logger.info(f"Balance loss value: {balance_loss.item():.4f}")

        # Combine all losses
        total_loss = (
            main_loss
            + self.uniformity_weight * uniformity_loss
            + self.balance_weight * balance_loss
        )

        if self.debug_logs:
            logger.info(f"\n8. Final Loss Components:")
            logger.info(f"Main loss: {main_loss.item():.4f}")
            logger.info(
                f"Weighted uniformity loss: {(self.uniformity_weight * uniformity_loss).item():.4f}"
            )
            logger.info(
                f"Weighted balance loss: {(self.balance_weight * balance_loss).item():.4f}"
            )
            logger.info(f"Total loss: {total_loss.item():.4f}")
            logger.info("\n=== End of Loss Computation ===\n")
            self.debug_logs = False  # Disable logs after first forward pass

        return total_loss


class AdaptiveLambdaStrategy(TrainingStrategy):
    """Training strategy that uses adaptive lambda weights for each head's CE loss."""

    def __init__(
        self,
        num_heads: int = 4,
        compare_with_final: bool = True,
        lambda_increase_rate: float = 1.1,
        lambda_decrease_rate: float = 0.9,
        min_lambda: float = 0.1,
        max_lambda: float = 10.0,
    ):
        """Initialize the adaptive lambda strategy."""
        self.num_heads = num_heads
        self.compare_with_final = compare_with_final
        self.lambda_increase_rate = lambda_increase_rate
        self.lambda_decrease_rate = lambda_decrease_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # Initialize lambda values to 1.0 for each head
        self.lambdas = torch.ones(num_heads)

        # Initialize accuracy tracking
        self.current_batch_correct = torch.zeros(num_heads)
        self.current_batch_total = 0
        self.debug_logs = True

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
        return nn.CrossEntropyLoss(reduction="none")  # Use 'none' to apply lambda weights per sample

    def update_accuracies(self, outputs: dict, target: torch.Tensor):
        """Update accuracy statistics for the current batch."""
        exit_names = sorted(outputs.keys())

        # Reset counters for new batch
        self.current_batch_correct.zero_()
        self.current_batch_total = target.size(0)

        # Calculate correct predictions for each head
        for i, exit_name in enumerate(exit_names):
            pred = outputs[exit_name].argmax(dim=1)
            self.current_batch_correct[i] = pred.eq(target).sum().item()

    def update_lambdas(self):
        """Update lambda values based on accuracy comparisons."""
        if self.current_batch_total == 0:
            return

        # Calculate accuracies
        accuracies = self.current_batch_correct / self.current_batch_total

        # Determine reference accuracies based on strategy
        if self.compare_with_final:
            reference_accuracy = accuracies[-1]  # Compare with final head
            for i in range(self.num_heads - 1):  # Skip the final head
                if accuracies[i] > reference_accuracy:
                    self.lambdas[i] *= self.lambda_decrease_rate
                else:
                    self.lambdas[i] *= self.lambda_increase_rate
        else:
            # Compare with next head
            for i in range(self.num_heads - 1):  # Skip the final head
                if accuracies[i] > accuracies[i + 1]:
                    self.lambdas[i] *= self.lambda_decrease_rate
                else:
                    self.lambdas[i] *= self.lambda_increase_rate

        # Clip lambda values
        self.lambdas.clamp_(self.min_lambda, self.max_lambda)

        if self.debug_logs:
            logger.info("\n=== Lambda Update ===")
            logger.info(f"Accuracies: {accuracies.tolist()}")
            logger.info(f"Updated lambdas: {self.lambdas.tolist()}")
            self.debug_logs = False

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        """Compute the weighted sum of CE losses."""
        # Update accuracy statistics
        self.update_accuracies(outputs, target)

        # Initialize loss
        loss = torch.tensor(0.0, device=target.device)

        # Sort exits by name to ensure consistent ordering
        exit_names = sorted(outputs.keys())

        # Compute weighted loss for each exit
        for i, exit_name in enumerate(exit_names):
            exit_loss = criterion(outputs[exit_name], target)  # Shape: [batch_size]
            weighted_loss = self.lambdas[i].to(target.device) * exit_loss
            loss += weighted_loss.mean()  # Take mean over batch

        # Update lambda values for next iteration
        self.update_lambdas()

        return loss


class LSECEStrategy(TrainingStrategy):
    """Training strategy that combines Log Sum Exp loss with Cross Entropy loss."""

    def __init__(self, beta: float = 10000.0):
        """Initialize the LSE-CE strategy."""
        self.beta = beta
        self.HEAD_DIM = 0
        self.BATCH_DIM = 1
        self.CLASS_DIM = 2

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(reduction="none")  # We need per-sample losses

    def compute_lse_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log sum exp loss for all heads."""
        return (1.0 / self.beta) * torch.logsumexp(
            self.beta * outputs, dim=self.CLASS_DIM
        ) - (1.0 / outputs.shape[self.CLASS_DIM])

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        """Compute the combined LSE-CE loss."""
        # Stack outputs to [num_heads, batch_size, num_classes] tensor
        exit_names = sorted(outputs.keys())
        stacked_outputs = torch.stack([outputs[name] for name in exit_names])

        # Compute CE loss for each head (shape: [num_heads, batch_size])
        ce_losses = []
        for output in stacked_outputs:
            head_loss = criterion(output, target)  # shape: [batch_size]
            ce_losses.append(head_loss)
        ce_loss = torch.stack(ce_losses)  # shape: [num_heads, batch_size]

        # Compute LSE loss (shape: [num_heads, batch_size])
        lse_loss = self.compute_lse_loss(stacked_outputs)

        # Combine losses with point-wise multiplication and take mean over batch
        combined_loss = (lse_loss * ce_loss).sum(dim=0).mean()

        return combined_loss


class FlattenedSoftmaxStrategy(TrainingStrategy):
    """Training strategy that applies softmax on flattened outputs and computes dot product loss."""

    def __init__(self):
        self.HEAD_DIM = 0
        self.BATCH_DIM = 1
        self.CLASS_DIM = 2

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        # We don't use a standard criterion as we implement our own loss
        return nn.Module()

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        # Stack outputs to [num_heads, batch_size, num_classes]
        exit_names = sorted(outputs.keys())
        stacked_outputs = torch.stack([outputs[name] for name in exit_names])
        num_heads, batch_size, num_classes = stacked_outputs.shape

        # Create a mask for target classes [batch_size, num_classes]
        target_mask = F.one_hot(target, num_classes=num_classes)

        # Expand dimensions to broadcast properly
        # [num_heads, batch_size, num_classes]
        target_mask = target_mask.unsqueeze(0).expand(num_heads, -1, -1)
        inverted_mask = 1 - target_mask

        # Get probabilities for target classes only
        # [num_heads, batch_size]
        target_probs = (stacked_outputs * target_mask).sum(dim=-1)
        no_target_probs = (stacked_outputs * inverted_mask).sum(dim=-1).mean()

        # Apply softmax across heads for each sample
        # [num_heads, batch_size]
        head_competition = F.softmax(target_probs, dim=0)

        # Compute 1 - Σ(sᵢ²) for each sample  
        # [batch_size]
        loss_per_sample = 1 - (head_competition**2).sum(dim=0)

        # Average over batch
        loss = loss_per_sample.mean() - torch.log(no_target_probs)
        return loss


class SingleHeadCEStrategy(TrainingStrategy):
    """Training strategy that assigns the true target to one random head per sample."""

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(reduction="none")

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        # Stack outputs to [num_heads, batch_size, num_classes]
        exit_names = sorted(outputs.keys())
        stacked_outputs = torch.stack([outputs[name] for name in exit_names])
        num_heads, batch_size, num_classes = stacked_outputs.shape  

        # Permute to [batch_size, num_heads, num_classes]
        stacked_outputs = torch.permute(stacked_outputs, (1, 0, 2))

        # Initialize losses tensor
        ce_losses = torch.zeros(num_heads, batch_size, device=target.device)

        # Create one-hot target matrix [batch_size, num_classes]
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()

        # For each head, create target matrix and compute loss
        for i in range(num_heads):
            # Create target matrix for current head
            target_matrix = torch.zeros(
                batch_size, num_heads, num_classes, device=target.device
            )
            target_matrix.fill_(1.0 / num_classes)  # Fill all elements with 1/num_classes
            target_matrix[:, i, :] = target_one_hot  # Override the i-th head with one-hot targets

            # Compute loss for current head
            ce_losses[i] = nn.functional.kl_div(
                torch.log_softmax(stacked_outputs, dim=-1),
                target_matrix,
                reduction="batchmean",
            )

        # Apply softmax to get probabilities
        head_probs = F.softmax(ce_losses, dim=0)

        # Compute Gini impurity: 1 - sum(p_i^2)
        gini_impurity = 1 - torch.sum(head_probs**2, dim=0)

        # Average over batch
        return gini_impurity.mean()


class MinHeadLossStrategy(TrainingStrategy):
    """Training strategy that uses the minimum loss across all heads."""

    def create_optimizer(
        self, model: nn.Module, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(reduction="none")  # We need per-sample losses

    def compute_loss(
        self, outputs: dict, target: torch.Tensor, criterion: nn.Module
    ) -> torch.Tensor:
        # Stack outputs to [num_heads, batch_size, num_classes]
        exit_names = sorted(outputs.keys())
        stacked_outputs = torch.stack([outputs[name] for name in exit_names])
        num_heads, batch_size, num_classes = stacked_outputs.shape

        # Initialize tensor to store losses for each head and sample
        head_losses = torch.zeros(num_heads, batch_size, device=target.device)

        # Compute loss for each head
        for i, exit_name in enumerate(exit_names):
            # Compute CE loss for current head (returns per-sample losses)
            head_losses[i] = criterion(outputs[exit_name], target)

        # Get minimum loss for each sample
        min_losses, min_indices = torch.min(head_losses, dim=0)

        # Create uniform distribution target
        uniform_target = torch.ones(batch_size, num_classes, device=target.device) / num_classes

        # Add KL divergence loss for non-selected heads
        kl_loss = torch.tensor(0.0, device=target.device)
        for i in range(num_heads):
            # Create mask for samples where this head was not selected
            not_selected_mask = (min_indices != i)
            if not_selected_mask.any():
                # Get outputs for this head for non-selected samples
                head_output = outputs[exit_names[i]][not_selected_mask]
                # Compute KL divergence with uniform distribution
                kl_loss += F.kl_div(
                    F.log_softmax(head_output, dim=1),
                    uniform_target[not_selected_mask],
                    reduction='batchmean'
                )

        # Combine minimum CE loss with KL divergence loss
        total_loss = min_losses.mean() + 0.1 * kl_loss  # 0.1 is a weighting factor

        return total_loss


def get_training_strategy(strategy_name: str, **kwargs) -> TrainingStrategy:
    """Factory function to get training strategy."""
    strategies = {
        "sum_ce": lambda: SumCEStrategy(),
        "kl_consistency": lambda: KLConsistencyStrategy(**kwargs),
        "multi_truth_penalty": lambda: MultiTruthPenaltyStrategy(**kwargs),
        "adaptive_lambda": lambda: AdaptiveLambdaStrategy(**kwargs),
        "lse_ce": lambda: LSECEStrategy(**kwargs),
        "flattened_softmax": lambda: FlattenedSoftmaxStrategy(),
        "sum_single_head_ce": lambda: SingleHeadCEStrategy(),
        "min_head_loss": lambda: MinHeadLossStrategy(),
    }
    if strategy_name not in strategies:
        raise ValueError(f"Unknown training strategy: {strategy_name}")
    return strategies[strategy_name]() 
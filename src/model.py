import torch
import torch.nn as nn
import torchvision.models as models
import logging
from abc import ABC, abstractmethod
import os

class EarlyExitModel(nn.Module, ABC):
    """Base class for early exit models that can be used for both vision and language tasks."""

    def __init__(self, num_classes: int, num_exits: int):
        """Initialize the early exit model.

        Args:
            num_classes (int): Number of output classes
            num_exits (int): Number of early exits (including final exit)
        """
        super(EarlyExitModel, self).__init__()
        self.num_classes = num_classes
        self.num_exits = num_exits
        self.softmax = nn.Softmax(dim=1)

    @abstractmethod
    def _make_aux_head(self, in_features: int, num_classes: int) -> nn.Module:
        """Create an auxiliary classification head.

        Args:
            in_features (int): Number of input features
            num_classes (int): Number of output classes

        Returns:
            nn.Module: The auxiliary classification head
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Dictionary containing outputs from all exit points
        """
        pass


class EarlyExitResNet(EarlyExitModel):
    FIXED_NUM_EXITS = 4  # 3 early exits + 1 final exit

    def __init__(
        self,
        num_classes: int,
        base_model: str = "resnet18",
        num_exits: int = FIXED_NUM_EXITS,
    ):
        if num_exits != self.FIXED_NUM_EXITS:
            raise ValueError(
                f"ResNet early exit model must have exactly {self.FIXED_NUM_EXITS} exits "
                f"(3 early exits + 1 final exit). Got {num_exits} exits instead."
            )
        super(EarlyExitResNet, self).__init__(num_classes, num_exits)

        # Load the base ResNet model
        if base_model == "resnet18":
            self.base_model = models.resnet18(weights=None, num_classes=num_classes)
        elif base_model == "resnet34":
            self.base_model = models.resnet34(weights=None, num_classes=num_classes)
        elif base_model == "resnet50":
            self.base_model = models.resnet50(weights=None, num_classes=num_classes)
        else:
            raise ValueError("Unsupported base model")

        # Get the number of features at each exit point
        features_exit1 = self.base_model.layer1[-1].conv2.out_channels
        features_exit2 = self.base_model.layer2[-1].conv2.out_channels
        features_exit3 = self.base_model.layer3[-1].conv2.out_channels

        # Define auxiliary heads
        self.aux_head1 = self._make_aux_head(features_exit1, num_classes)
        self.aux_head2 = self._make_aux_head(features_exit2, num_classes)
        self.aux_head3 = self._make_aux_head(features_exit3, num_classes)
        
    def _make_aux_head(self, in_features: int, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Copied from torchvision.models.resnet, just added the aux heads
        # First block
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)

        exit1 = self.aux_head1(x)

        # Second block
        x = self.base_model.layer2(x)
        exit2 = self.aux_head2(x)

        # Third block
        x = self.base_model.layer3(x)
        exit3 = self.aux_head3(x)

        # Final block
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        final = self.base_model.fc(x)
        final = self.softmax(final)

        return {"exit1": exit1, "exit2": exit2, "exit3": exit3, "final": final}


# Function to create the model
def create_early_exit_resnet(num_classes, base_model="resnet18"):
    return EarlyExitResNet(num_classes, base_model)


def load_model(model_path: str, num_classes: int, base_model: str) -> torch.nn.Module:
    """Load the early exit model from checkpoint.

    Args:
        model_path (str): Path to the model checkpoint or directory
        num_classes (int): Number of classes in the dataset
        base_model (str): Base model architecture name

    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    try:
        # If model_path is a directory, look for best_model.pt inside
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "best_model.pt")
            
        model = create_early_exit_resnet(num_classes=num_classes, base_model=base_model)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        logging.info("Successfully loaded model from %s", model_path)
        return model
    except FileNotFoundError:
        logging.error("Trained model not found at %s", model_path)
        raise
    except Exception as e:
        logging.error("Failed to load model: %s", str(e))
        raise

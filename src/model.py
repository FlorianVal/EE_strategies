import torch
import torch.nn as nn
import torchvision.models as models
import logging
from abc import ABC, abstractmethod


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
            self.base_model = models.resnet18(weights=None)
        elif base_model == "resnet34":
            self.base_model = models.resnet34(weights=None)
        elif base_model == "resnet50":
            self.base_model = models.resnet50(weights=None)
        else:
            raise ValueError("Unsupported base model")

        # Get the number of features at each exit point
        self.features_exit1 = self.base_model.layer1[-1].conv2.out_channels
        self.features_exit2 = self.base_model.layer2[-1].conv2.out_channels
        self.features_exit3 = self.base_model.layer3[-1].conv2.out_channels
        self.features_final = self.base_model.fc.in_features

        # Remove the final fully connected layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # Define auxiliary heads
        self.aux_head1 = self._make_aux_head(self.features_exit1, num_classes)
        self.aux_head2 = self._make_aux_head(self.features_exit2, num_classes)
        self.aux_head3 = self._make_aux_head(self.features_exit3, num_classes)

        # Final classification layers
        self.final_bn = nn.BatchNorm1d(self.features_final)
        self.final_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.features_final, num_classes)

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
        # First block
        x = self.base_model[0](x)  # Conv1
        x = self.base_model[1](x)  # BN1
        x = self.base_model[2](x)  # ReLU
        x = self.base_model[3](x)  # MaxPool
        x = self.base_model[4](x)  # Layer1
        exit1 = self.aux_head1(x)

        # Second block
        x = self.base_model[5](x)  # Layer2
        exit2 = self.aux_head2(x)

        # Third block
        x = self.base_model[6](x)  # Layer3
        exit3 = self.aux_head3(x)

        # Final block
        x = self.base_model[7](x)  # Layer4
        x = self.base_model[8](x)  # AvgPool
        x = torch.flatten(x, 1)
        x = self.final_bn(x)
        x = self.final_dropout(x)
        final = self.fc(x)
        final = self.softmax(final)

        return {"exit1": exit1, "exit2": exit2, "exit3": exit3, "final": final}


# Function to create the model
def create_early_exit_resnet(num_classes, base_model="resnet18"):
    return EarlyExitResNet(num_classes, base_model)


def load_model(model_path: str, num_classes: int, base_model: str) -> torch.nn.Module:
    """Load the early exit model from checkpoint.

    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of classes in the dataset
        base_model (str): Base model architecture name

    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    try:
        model = create_early_exit_resnet(num_classes=num_classes, base_model=base_model)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        logging.info("Successfully loaded model from %s", model_path)
        return model
    except FileNotFoundError:
        logging.error("Trained model not found at %s", model_path)
        raise
    except Exception as e:
        logging.error("Failed to load model: %s", str(e))
        raise


# Additional helper functions or classes can be added here

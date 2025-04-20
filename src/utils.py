# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random

class Accuracy:
    """
    This class is the accuracy object.

     Attributes:
        correct: The number of correct predictions.
        total: The total number of examples to classify.
    """

    correct: int
    total: int

    def _init_(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions: [batch, number of classes]
            labels: labels of the examples. Dimensions: [batch].
        """

        # Compute predictions
        predictions = logits.argmax(1).type_as(labels)

        # Update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            float: The computed accuracy as the ratio of correct predictions to total examples.
        """

        return self.correct / self.total

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    # model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    # model_scripted.save(f"models/{name}.pt")

    # Definir ruta completa al archivo
    path = os.path.join("models", f"{name}.pt")

    # Guardar solo los pesos del modelo
    torch.save(model.state_dict(), path)

    print(f"✅ Modelo guardado en '{path}'")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None



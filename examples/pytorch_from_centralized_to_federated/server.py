"""Flower server example."""

from collections import OrderedDict


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.common import Metrics

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
import flwr as fl
import os
import cifar
import torch
import torchvision
from typing import Callable, Optional, Tuple, Dict, Union
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch import Tensor
import cifar

def get_evaluate_fn(
    testset: torchvision.datasets.MNIST,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = cifar.Net().to(device).eval()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
       

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = cifar.test(model, testloader, device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    DATA_ROOT_SERVER = "./server_dataset"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    testset = MNIST(DATA_ROOT_SERVER, train=False, download=True, transform=transform)
        # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

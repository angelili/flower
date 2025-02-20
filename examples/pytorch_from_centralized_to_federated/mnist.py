"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""




from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import MNIST
import random
import torch


DATA_ROOT = "./dataset"
Dominant_class=False
Non_uniform_cardinality=True

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    # Load the MNIST dataset
    trainset = MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    if Dominant_class==True:
  

        # Define the class to include 80% of samples
        class_to_include = random.randint(0, 9)  # Change this to the desired class

        # Get the indices of samples belonging to the specified class
        class_indices = torch.where(trainset.targets == class_to_include)[0]

        # Convert to np.array
        class_indices=class_indices.numpy()

        # Calculate the number of samples to include from the specified class
        num_samples_class = int(len(class_indices) * 0.8)

        # Shuffle the set of samples from the specified class
        np.random.shuffle(class_indices)

        # Select the first `num_samples_class` elements
        subset_indices_class = class_indices[:num_samples_class]

        # Convert back to tensor
        subset_indices_class=torch.from_numpy(subset_indices_class)

        # Create a Subset of the original dataset using the selected subset indices from the specified class
        subset_dataset_class = torch.utils.data.Subset(trainset, subset_indices_class)

        # Calculate the number of samples to include from other classes
        num_samples_other = int(len(trainset) * 0.2)

        # Get the indices of samples from other classes
        other_indices = torch.where(trainset.targets != class_to_include)[0]

        # Convert to np.array
        other_indices=other_indices.numpy()

        # Shuffle the set of samples from other classes
        np.random.shuffle(other_indices)

        # Select the first `num_samples_others` elements
        subset_indices_other = other_indices[:num_samples_class]

        # Convert back to tensor
        subset_indices_other=torch.from_numpy(subset_indices_other)

        # Create a Subset of the original dataset using the selected subset indices from other classes
        subset_dataset_other = torch.utils.data.Subset(trainset, subset_indices_other)

        # Concatenate the subsets from the specified class and other classes
        trainset = torch.utils.data.ConcatDataset([subset_dataset_class, subset_dataset_other])

        # Create the DataLoader with the specified subsets
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)




        # Define the class to include 80% of samples
        class_to_include = random.randint(0, 9)  # Change this to the desired class

        # Get the indices of samples belonging to the specified class
        class_indices = torch.where(testset.targets == class_to_include)[0]

        # Convert to np.array
        class_indices=class_indices.numpy()

        # Calculate the number of samples to include from the specified class
        num_samples_class = int(len(class_indices) * 0.8)

        # Shuffle the set of samples from the specified class
        np.random.shuffle(class_indices)

        # Select the first `num_samples_class` elements
        subset_indices_class = class_indices[:num_samples_class]

        # Convert back to tensor
        subset_indices_class=torch.from_numpy(subset_indices_class)

        # Create a Subset of the original dataset using the selected subset indices from the specified class
        subset_dataset_class = torch.utils.data.Subset(testset, subset_indices_class)

        # Calculate the number of samples to include from other classes
        num_samples_other = int(len(testset) * 0.2)

        # Get the indices of samples from other classes
        other_indices = torch.where(testset.targets != class_to_include)[0]

        # Convert to np.array
        other_indices=other_indices.numpy()

        # Shuffle the set of samples from other classes
        np.random.shuffle(other_indices)

        # Select the first `num_samples_others` elements
        subset_indices_other = other_indices[:num_samples_class]

        # Convert back to tensor
        subset_indices_other=torch.from_numpy(subset_indices_other)

        # Create a Subset of the original dataset using the selected subset indices from other classes
        subset_dataset_other = torch.utils.data.Subset(testset, subset_indices_other)

        # Concatenate the subsets from the specified class and other classes
        testset = torch.utils.data.ConcatDataset([subset_dataset_class, subset_dataset_other])

        # Create the DataLoader with the specified subsets
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}

        return trainloader, testloader, testset, num_examples
    
    if Non_uniform_cardinality==True:
        sample_size_train = random.randint(200, 1000)
        sample_size_test =  int(sample_size_train*0.1)
    else:
        sample_size_train=500
        sample_size_test=100

    indices_train = random.sample(range(len(trainset)), sample_size_train)
    sampler_train= torch.utils.data.SubsetRandomSampler(indices_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, sampler=sampler_train)
    print(len(trainloader))
    print('dugi',sample_size_train)
    print('kraci', sample_size_test)
    indices_test = random.sample(range(len(testset)), sample_size_test)
    sampler_test = torch.utils.data.SubsetRandomSampler(indices_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, sampler=sampler_test)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}

    return trainloader, testloader, testset, num_examples


# pylint: disable=unsubscriptable-object


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4* 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _, _ = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()

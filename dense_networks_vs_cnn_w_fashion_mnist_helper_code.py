### SETUP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import random
import time
from datetime import datetime

from typing import Dict

### STANDARD IPYWIDGET IMPORTS

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output

from IPython.display import display, clear_output

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### LOAD DATASET

# FASHION MNIST

# mean and std_dev for Fashion MNIST
# note that these are different from those for the standard Numeric MNIST Dataset

fashion_mean = 0.2860
fashion_std_dev = 0.3530

fashion_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(fashion_mean, fashion_std_dev)
])

def display_load_dataset():
    global fashion_training_images, fashion_testing_images, \
            fashion_training_images_length, fashion_testing_images_length

    fashion_training_images = datasets.FashionMNIST("FashionMNIST_data", transform=fashion_transform,
                                            download=True,train=True)
    fashion_testing_images = datasets.FashionMNIST("FashionMNIST_data", transform=fashion_transform,
                                            download=True,train=False)
    fashion_training_images_length = len(fashion_training_images)
    fashion_testing_images_length = len(fashion_testing_images)

class MNISTBasicNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, network_input):
        return self.layers(network_input)  # Return raw logits

    def predict_with_softmax(self, inputs):
        """
        Perform a forward pass and apply softmax to get probabilities.
        """
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            logits = self.forward(inputs)  # Get raw logits
            probabilities = nn.functional.softmax(logits, dim=1)  # Convert to probabilities
        return probabilities

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 64 feature maps of size 7x7 -> 128 units
            nn.ReLU(),
            nn.Linear(128, 10)  # 128 -> 10 output classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def predict_with_softmax(self, inputs):
        """
        Perform a forward pass and apply softmax to get probabilities.
        """
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            logits = self.forward(inputs)  # Get raw logits
            probabilities = nn.functional.softmax(logits, dim=1)  # Convert to probabilities
        return probabilities
    
class ImprovedFashionMNISTCNN(nn.Module):
    def __init__(self):
        super(ImprovedFashionMNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: (1, 28, 28), Output: (32, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # Output: (32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, 14, 14)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Output: (64, 7, 7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: (128, 7, 7)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # Output: (128, 3, 3)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    
def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def display_define_and_display_networks():

    dense_network = MNISTBasicNetwork()  # local and only used for printing
    cnn_network = ImprovedFashionMNISTCNN()

    display(HTML("<h3>Dense Neural Network Architecture</h3>"))
    print(dense_network)
    print(f"Total Parameters: {count_parameters(dense_network):,}")
    display(HTML("<h3>Convolutional Neural Network Architecture</h3>"))
    print(cnn_network)
    print(f"Total Parameters: {count_parameters(cnn_network):,}")

# HYPERPARAMETERS
NUMBER_OF_EPOCHS = 8
LEARNING_RATE = 0.02
BATCH_SIZE = 128
PRINT_RATE = 1

def train_network(model, num_epochs):    
    training_loader = DataLoader(fashion_training_images,
                                    batch_size=BATCH_SIZE, shuffle=True)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    model.train()

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        total_instances = 0
        total_loss = 0
        model.train()
        for img_batch, label_batch in training_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            result_batch = model(img_batch)
            loss = loss_func(result_batch, label_batch)

            mini_batch_size = len(img_batch)
            total_instances += mini_batch_size
            total_loss += mini_batch_size * loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % PRINT_RATE == 0:
            print(f"epoch: {epoch}, loss: {(total_loss / total_instances):.4f}")

    end_time = time.time()
    print(f"Training time: {(end_time - start_time):.2f} seconds")

def determine_accuracy(model):
    test_loader = DataLoader(fashion_testing_images, batch_size=BATCH_SIZE, shuffle=False)

    total_correct = 0
    model.eval()
    with torch.no_grad():
        for img_batch, label_batch in test_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            result_batch = model(img_batch)
            _, predicted_batch = torch.max(result_batch, 1)
            total_correct += (predicted_batch == label_batch).sum().item()

    total_count = len(test_loader.dataset)
    return (100 * total_correct / total_count, total_correct, total_count)

def print_accuracy(model):
    """
    Prints the accuracy, the number of correct predictions, and the total number of samples.
    Returns the results as a tuple: (accuracy, total_correct, total_samples).
    """
    results = determine_accuracy(model)
    accuracy, correct, total = results
    print(f"Correctly Predicted: {correct:,}")
    print(f"Total Samples: {total:,}")
    print(f"Accuracy: {accuracy:.2f}%")
    return results

def display_create_and_train_dense_network():
    global dense_network, dense_accuracy

    dense_network = MNISTBasicNetwork()
    train_network(dense_network,NUMBER_OF_EPOCHS)
    (dense_accuracy,_,_) = print_accuracy(dense_network)

def display_create_and_train_cnn_network():
    global cnn_network, cnn_accuracy

    cnn_network = ImprovedFashionMNISTCNN()
    train_network(cnn_network,NUMBER_OF_EPOCHS)
    (cnn_accuracy,_,_) = print_accuracy(cnn_network)

def display_counts():
    display(HTML("<h3>Dense Neural Network Architecture</h3>"))
    print(count_parameters_detailed(dense_network))

    display(HTML("<h3>Convolutional Neural Network Architecture</h3>"))
    print(count_parameters_detailed(cnn_network))

def display_relative_accuracy():
    print(f"Dense Accuracy: {dense_accuracy}%")
    print(f"CNN Accuracy: {cnn_accuracy}%")
    relative_improvement = 100 * (cnn_accuracy - dense_accuracy) / (100 - dense_accuracy)
    display(HTML(f"<b>Relative Improvement: {relative_improvement:.2f}%</b>"))
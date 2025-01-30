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
from ipywidgets import HBox, VBox, HTML, Button, Output, Layout

from IPython.display import clear_output

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

# Fashion MNIST

# mean and std_dev for Fashion MNIST
# note that these are different from those for the standard Numeric MNIST Dataset
fashion_mean = 0.2860
fashion_std_dev = 0.3530

fashion_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(fashion_mean, fashion_std_dev)
])

# Regular MNIST values
regular_mean = 0.13066048920154572
regular_std_dev = 0.308107852935791

regular_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(regular_mean, regular_std_dev)
])

# In function so students will see load progress when they run "Load Dataset" Cell
def display_load_dataset():
    # Fashion MNIST
    global fashion_training_images, fashion_testing_images


    fashion_training_images = datasets.FashionMNIST("FashionMNIST_data", transform=fashion_transform,
                                            download=True,train=True)
    fashion_testing_images = datasets.FashionMNIST("FashionMNIST_data", transform=fashion_transform,
                                            download=True,train=False)

    # Regular MNIST
    global regular_training_images, regular_testing_images

    regular_training_images = datasets.MNIST("MNIST_data", transform=regular_transform,
                                            download=True,train=True)
    regular_testing_images = datasets.MNIST("MNIST_data", transform=regular_transform,
                                            download=True,train=False)
    
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
    
def to_fashion_string(class_id):
    fashion_mnist_classes = [
        "T-shirt/Top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]
    return fashion_mnist_classes[class_id]

# Display Samples
def regenerate_explore_samples_data(_, output, dataset, curr_mean, curr_std_dev, convert_flag):
    ROWS = 4
    COLS = 4

    with output:
        output.clear_output(wait=True)

        fig, axs = plt.subplots(4, 4)
        for row in range(ROWS):
            for col in range(COLS):
                image, label = dataset[random.randint(0, len(dataset) - 1)]
                if convert_flag: 
                    output_label = f"{label}:{to_fashion_string(label)}"
                else:
                    output_label = f"{label}"
                denormalized_image = image.squeeze() * curr_std_dev + curr_mean
                axs[row][col].imshow(denormalized_image, cmap='gray')  # squeeze to remove unnecessary dimension
                axs[row][col].text(2, 2, str(output_label), color='yellow', fontsize=12, ha='left', va='top')
                axs[row][col].axis('off')
        
        plt.tight_layout()
        plt.show()


# Create fashion button and output widget
explore_display_fashion_samples_button = Button(description="Display More Samples", button_style='info',
                                            layout=Layout(width="180px"))
explore_fashion_samples_output = Output()


# Attach the function to the button click
explore_display_fashion_samples_button.on_click(
    lambda _: regenerate_explore_samples_data(_, explore_fashion_samples_output, fashion_training_images,
                                              fashion_mean, fashion_std_dev, True))

# Display the button and output
# display(html_style,VBox([explore_display_samples_button, explore_samples_output]))

def display_explore_fashion_data():
    display(explore_display_fashion_samples_button,explore_fashion_samples_output)
    regenerate_explore_samples_data(None, explore_fashion_samples_output, fashion_training_images,
                                              fashion_mean, fashion_std_dev, True)

# Create regular button and output widget
explore_display_regular_samples_button = Button(description="Display More Samples", button_style='info',
                                            layout=Layout(width="180px"))
explore_regular_samples_output = Output()


# Attach the function to the button click
explore_display_regular_samples_button.on_click(
    lambda _: regenerate_explore_samples_data(_, explore_regular_samples_output, regular_training_images,
                                              regular_mean, regular_std_dev, False))

# Display the button and output
# display(html_style,VBox([explore_display_samples_button, explore_samples_output]))

def display_explore_regular_data():
    display(explore_display_regular_samples_button,explore_regular_samples_output)
    regenerate_explore_samples_data(None, explore_regular_samples_output, regular_training_images,
                                              regular_mean, regular_std_dev, False)


NUMBER_OF_EPOCHS = 8
LEARNING_RATE = 0.02
BATCH_SIZE = 128

PRINT_RATE = 1  # how often we should print results

def train_network(model, train_dataset):
    training_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE)
    model.train()
    
    start_time = time.time()
    
    for epoch in range(1,NUMBER_OF_EPOCHS+1):
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
            print(f"epoch: {epoch}, loss: {(total_loss/total_instances):.4f}")
    
    end_time = time.time()
    print(f"Training time: {(end_time - start_time):.2f} seconds")
    
def determine_accuracy(model, test_loader):

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

def print_accuracy(model, testing_dataset):
    testing_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True) 

    accuracy, correct, total = determine_accuracy(model, testing_loader)
    print(f"Correctly Predicted: {correct:,}")
    print(f"Total Samples: {total:,}")
    print(f"Accuracy: {accuracy:.2f}%")

def regular_mnist_create_train_test():
    regular_network = MNISTBasicNetwork()
    regular_network.to(device)

    display(HTML("<b>Regular MNIST</b>"))
    train_network(regular_network, regular_training_images)
    print_accuracy(regular_network, regular_testing_images)
    
    return regular_network

def display_train_regular_mnist():
    global regular_network
    regular_network = regular_mnist_create_train_test()

def fashion_mnist_create_train_test():
    fashion_network = MNISTBasicNetwork()
    fashion_network.to(device)

    display(HTML("<b>Fashion MNIST</b>"))
    train_network(fashion_network, fashion_training_images)
    print_accuracy(fashion_network, fashion_testing_images)
    
    return fashion_network

def display_train_fashion_mnist():
    global fashion_network
    fashion_network = fashion_mnist_create_train_test()

def display_network_architectures():
    display(HTML("<b>Regular MNIST</b>"))
    print(regular_network)
    display(HTML("<b>Fashion MNIST</b>"))
    print(fashion_network)
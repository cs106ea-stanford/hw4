from torchvision import models, datasets, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import io
import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
import random
import copy

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load ResNet18 with pre-trained weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
model.eval()  # Set to evaluation mode

# Define preprocessing pipeline for the image
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize shortest side to 256
    transforms.CenterCrop(224),            # Center crop to 224x224
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],        # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
])

def visualize_feature_maps(image):
    """
    Visualizes feature maps of the first and last convolutional layers of ResNet18.
    """
    # Define hooks to capture feature maps
    first_layer_features = []
    last_layer_features = []

    def hook_first_layer(module, input, output):
        first_layer_features.append(output.to(device))  # Ensure output is on the same device

    def hook_last_layer(module, input, output):
        last_layer_features.append(output.to(device))  # Capture last CNN layer before pooling

    # Register hooks
    model.conv1.register_forward_hook(hook_first_layer)  # First conv layer
    model.layer4.register_forward_hook(hook_last_layer)  # Last CNN layer before pooling

    # Preprocess image
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Run the model
    with torch.no_grad():
        model(input_tensor)

    # ---- First Layer Visualization ----
    first_layer_map = first_layer_features[0][0]  # Get first batch
    num_channels_first = first_layer_map.shape[0]

    # Normalize for visualization
    first_layer_map = (first_layer_map - first_layer_map.min()) / (first_layer_map.max() - first_layer_map.min())

    # Plot first layer feature maps
    fig, axs = plt.subplots(1, min(8, num_channels_first), figsize=(12, 2))
    for i in range(min(8, num_channels_first)):
        axs[i].imshow(first_layer_map[i].cpu().detach().numpy(), cmap='viridis')
        axs[i].axis('off')

    plt.suptitle("First Layer Feature Maps", fontsize=14)
    plt.tight_layout(pad=0.5)
    plt.show()

    # ---- Last Layer (CNN) Visualization ----
    last_layer_map = last_layer_features[0][0]  # Get first batch
    num_channels_last = last_layer_map.shape[0]  # Should be 512
    
    # Print shape to verify we're handling it correctly
    print(f"Last layer feature map shape: {last_layer_map.shape}")  # Should be (512, 7, 7)
    
    # Normalize for visualization
    last_layer_map = (last_layer_map - last_layer_map.min()) / (last_layer_map.max() - last_layer_map.min())
    
    # Last layer feature maps (only showing 8 for readability)
    fig, axs = plt.subplots(2, 4, figsize=(8, 4), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})  
    axs = axs.flatten()
    
    for i in range(8):  # Display only 8 out of 512
        img = last_layer_map[i].cpu().detach().numpy()
        axs[i].imshow(img, cmap='viridis', interpolation='nearest')
        axs[i].axis('off')
    
    plt.suptitle("Last CNN Layer Feature Maps (Before Pooling)", fontsize=14)
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.05, hspace=0.05)  # Tighten layout
    plt.show()


def classify_image(_, output, upload_widget, visualize_flag):
    with output:
        output.clear_output()  # Clear previous output

        # Check if a file has been uploaded
        if not upload_widget.value:
            print("No file uploaded.")
            return

        # Ensure the value is always a tuple 
        if not isinstance(upload_widget.value, tuple):
            upload_value = tuple(upload_widget.value.values())
        else:
            upload_value = upload_widget.value

        # Get the first (and only) file in the tuple
        uploaded_file = upload_value[0]
        image_data = uploaded_file['content']

        # Open the image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Display Original Image
        print("Original Image:")
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

        # Apply cropping separately to preview it
        cropped_image = transforms.CenterCrop(224)(transforms.Resize(256)(image))

        # Display Cropped Image
        print("Cropped to 224x224:")
        plt.figure(figsize=(3, 3))
        plt.imshow(cropped_image)
        plt.axis("off")
        plt.show()

        if visualize_flag:
            visualize_feature_maps(image)
    
        # Preprocess the image
        input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    
        # Run the model
        with torch.no_grad():
            output = model(input_tensor)
    
        # Get predicted probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        # Map the top 5 indices to human-readable labels
        labels = models.ResNet18_Weights.DEFAULT.meta['categories']
        top5_labels = [labels[idx] for idx in top5_idx]

        # Display the results
        print("Top 5 Predictions:")
        for i in range(5):
            print(f"{i+1}: {top5_labels[i]} - {top5_prob[i].item() * 100:.2f}%")

class_output = widgets.Output()
# Create an image upload widget
basic_upload_widget = widgets.FileUpload(
    accept='image/*',  # Accept only image files
    multiple=False     # Single file at a time
)

# Attach the function to the widget
basic_upload_widget.observe(lambda _: classify_image(_, class_output, basic_upload_widget, False), names='value')

def display_classification_upload():
    display(basic_upload_widget, class_output)

kernels_output = widgets.Output()
# Create an image upload widget
kernels_upload_widget = widgets.FileUpload(
    accept='image/*',  # Accept only image files
    multiple=False     # Single file at a time
)

# Attach the function to the widget
kernels_upload_widget.observe(lambda _: classify_image(_, kernels_output, kernels_upload_widget, True), names='value')

def display_kernels_upload():
    display(kernels_upload_widget, kernels_output)


def display_resnet18_architecture():
    display(widgets.HTML("<b>ResNet18 Architecture</b>"))
    print(model)
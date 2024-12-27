import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import streamlit as st

# --- BasicCNN Model Definition ---
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Helper functions ---
def preprocess_image(img_path, transform):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def show_cam_on_image(img, mask):
    # Ensure the image and mask (heatmap) have the same size
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def deprocess(image):
    image = image.clone()
    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    image = np.uint8(image * 255)
    return image

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer, model_type):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model_type = model_type

        # Register hooks to save gradients and activations
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_cam(self, image, target_class=1):
        self.model.zero_grad()
        output = self.model(image)
        output[0, target_class].backward()  # Gradient for target class
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        # If gradient is empty then exit
        if gradients.size == 0:
            print('Gradient is empty, please select another layer')
            st.error('Gradient is empty, please select another layer')
            return None

        if activations.size == 0:
            print('Activation is empty, please select another layer')
            st.error('Activation is empty, please select another layer')
            return None

        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(activations * weights, axis=1, keepdims=True)
        cam = np.maximum(cam, 0)  # ReLU
        return cam[0, 0]  # return only the 2D array heatmap

# --- Model Loading and Setup ---
def get_model(model_type, device):
    model = BasicCNN()  # Using BasicCNN model
    model.load_state_dict(torch.load("./saved_models/best_cnn_model.pth", map_location=device))
    target_layer = model.conv2  # Choosing conv2 layer for Grad-CAM
    return model, target_layer

# --- Streamlit Interface ---
st.title('Grad-CAM for Skin Lesion Classification')

# File upload for the image
uploaded_file = st.file_uploader("Upload a Skin Lesion Image", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Ensure the uploaded_images directory exists
    uploaded_images_dir = './uploaded_images'
    if not os.path.exists(uploaded_images_dir):
        os.makedirs(uploaded_images_dir)

    # Save the uploaded image
    image_path = os.path.join(uploaded_images_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_size = 224
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_type = "cnn"  # Specify model type (BasicCNN here)
    model, target_layer = get_model(model_type, device)
    model.to(device)
    grad_cam = GradCAM(model, target_layer, model_type)

    # Preprocess the uploaded image
    image = preprocess_image(image_path, transform).to(device)

    # Apply Grad-CAM
    cam = grad_cam.get_cam(image)
    if cam is not None:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) #convert to RGB
        original_image = cv2.resize(original_image, (image_size, image_size))
        cam = cv2.resize(cam, (image_size, image_size)) #Resize the heatmap
        cam = cam / np.max(cam)  # Normalize after resizing
        cam_image = show_cam_on_image(original_image, cam)

        # Display the Grad-CAM image
        st.image(cam_image, caption="Grad-CAM Result", use_column_width=True)
    else:
        st.write("Error: Could not generate Grad-CAM heatmap.")
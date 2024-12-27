# Example Streamlit app
You can find the streamlit app here 
for prediction:
'https://ridhwan-skin-lesion-app.streamlit.app/'

for gradcam
'https://ridhwan-skinlesion-gradcam.streamlit.app/'

# Streamlit Image and Metadata Model App

This project demonstrates how to build a Streamlit app that works with an image-only model and a metadata model for predictions. The app is designed to accept an image and its associated metadata (such as labels or other descriptors) to make predictions.

## Prerequisites

- **Python 3.10.12** (ensure you're using the correct version)
- **Conda** (to manage your environment) i use miniconda in this case

## Installation and Setup

Follow these steps to set up the environment and run the Streamlit app locally.

### 1. Create a Conda Environment

First, create a Conda environment with Python 3.10.12:

```bash
conda create -n streamlit-env python=3.10.12
conda activate streamlit-env
```

2. Install Dependencies
You can install all the required dependencies using the requirements.txt file provided:

```bash command
pip install -r requirements.txt
```

This will install all necessary libraries, including streamlit, torch, pandas, numpy, and others used by the app. You can find the requirements.txt in this repo

3. Running the Streamlit App
Once your environment is set up and the dependencies are installed, you can run the Streamlit app:

```bash
streamlit run app.py
```

This will launch the Streamlit app in your browser, where you can interact with it by uploading images and associated metadata.

## Image-Only Model vs. Metadata Model

In this project, you will work with two types of models:

### **Image-Only Model**

An image-only model focuses solely on the **visual features** of an image. 

**Handling the Image-Only Model:**

*   **Input:** Only the image itself (e.g., JPEG, PNG).
*   **Output:** Predicted label

### **Metadata Model**

A metadata model uses **additional information** alongside the image. This could include numerical, categorical, or textual data that is related to the image but not directly extracted from it.

**Handling the Metadata Model:**

*   **Input:** Both the image and metadata (e.g., text descriptions, numerical features).
*   **Process:** The model processes the image using traditional methods (e.g., CNNs), and the metadata is either concatenated with the image features or passed separately through a different network layer before making predictions.
*   **Output:** A prediction that takes into account both the **visual features** and the **associated metadata**.

For metadata, the Streamlit app allows users to input additional information (such as a description or label) in fields that are passed to the model during inference.
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# --- Model and Transforms ---
image_size = 224
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load the TorchScript model
def load_traced_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)  # Load TorchScript traced model
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to process metadata
def process_metadata(age, sex, anatom_site):
    sex_categories = ['male', 'female']
    anatom_categories = ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']

    def _one_hot_encode(value, categories):
        encoding = [0.0] * len(categories)  # default all 0.0
        if value in categories:
            idx = categories.index(value)  # find index of value
            encoding[idx] = 1.0  # then 1 hot
        return encoding

    metadata_values = []
    metadata_values.extend(_one_hot_encode(sex, sex_categories))
    metadata_values.extend(_one_hot_encode(anatom_site, anatom_categories))
    metadata_values.append(float(age))

    metadata = np.array(metadata_values, dtype=np.float32)  # Convert metadata into numpy first
    metadata = torch.tensor(metadata, dtype=torch.float32)  # Then convert metadata into tensor
    return metadata

# Main Streamlit app
def main():
    st.title("Skin Lesion Classification App")
    st.write("Upload a skin lesion image for classification.")

    # Choose model
    model_type = st.selectbox("Select Model Type", ["cnn", "efficientnet", "cnn_metadata"])

    # Model paths
    model_paths = {
        "cnn": "best_cnn_model_traced.pt", 
        "efficientnet": "best_efficientnet_model_traced.pt",
        "cnn_metadata": "best_cnn_metadata_model_traced.pt"
    }
    model_path = model_paths.get(model_type, None)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Get metadata input
    if model_type == 'cnn_metadata':
        st.sidebar.header("Metadata Input")
        age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=40)
        sex = st.sidebar.selectbox("Sex", ["male", "female"])
        anatom_site = st.sidebar.selectbox("Anatomical Site", ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital'])

    if uploaded_file is not None:
        try:
            # Open the image with PIL
            image = Image.open(uploaded_file).convert('RGB')

            # Display the uploaded image
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            # Preprocess the image
            input_image = val_transform(image).unsqueeze(0)  # Add batch dimension
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model
            model = load_traced_model(model_path, device)

            if model_type == "cnn_metadata":
                metadata = process_metadata(age, sex, anatom_site)
                metadata = metadata.unsqueeze(0).to(device)  # Ensure metadata is also on the correct device
                with torch.no_grad():
                    output = model(input_image.to(device), metadata)  # Pass both image and metadata
            else:
                with torch.no_grad():
                    output = model(input_image.to(device))  # Only pass the image for non-metadata models

            # Make prediction
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            class_names = ["Benign", "Malignant"]
            predicted_label = class_names[predicted_class]

            # Display prediction
            st.write("## Prediction Results:")
            st.write(f"Predicted Class: **{predicted_label}**")
            st.write(f"Confidence: **{confidence:.4f}**")
            st.write(f"Probability (Benign): {probabilities[0]:.4f}")
            st.write(f"Probability (Malignant): {probabilities[1]:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an image for prediction.")

if __name__ == "__main__":
    main()

import os
import time
import cv2
import h5py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# DnCNN Model
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning


# Load weights
def load_h5_weights(mat_file_path, model):
    print(f"Loading weights from: {mat_file_path}")
    with h5py.File(mat_file_path, 'r') as f:
        weights_datasets = []
        biases_datasets = []
        for key in f.keys():
            if key.startswith("#refs#"):
                for subkey in f[key]:
                    obj = f[f"{key}/{subkey}"]
                    if isinstance(obj, h5py.Dataset):
                        if len(obj.shape) == 4:
                            weights_datasets.append(obj)
                        elif len(obj.shape) == 2:
                            biases_datasets.append(obj)

        with torch.no_grad():
            weight_idx = 0
            bias_idx = 0
            for layer in model.dncnn:
                if isinstance(layer, nn.Conv2d):
                    if weight_idx < len(weights_datasets):
                        weight_data = weights_datasets[weight_idx][()]
                        if weight_data.shape == tuple(layer.weight.shape):
                            layer.weight.copy_(torch.tensor(weight_data))
                        weight_idx += 1
                    if layer.bias is not None and bias_idx < len(biases_datasets):
                        bias_data = biases_datasets[bias_idx][()]
                        if bias_data.shape == tuple(layer.bias.shape):
                            layer.bias.copy_(torch.tensor(bias_data))
                        bias_idx += 1
    print("Weights successfully loaded into the model!")


# Denoising
def denoise_with_cnn(model, noisy_image_path):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    if noisy_image is None:
        raise FileNotFoundError(f"Noisy image not found: {noisy_image_path}")
    noisy_image = noisy_image / 255.0
    noisy_tensor = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze(0).squeeze(0).numpy()
    denoised_image = (denoised_tensor * 255).astype(np.uint8)
    return denoised_image


# Post-Processing
def post_process_document(denoised_image):
    thresholded = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 25, 2)
    blended = cv2.addWeighted(thresholded, 0.8, denoised_image, 0.2, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_image = cv2.morphologyEx(blended, cv2.MORPH_OPEN, kernel)
    return final_image


# Save PDF
def save_as_pdf(image_path, pdf_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image.save(pdf_path, "PDF", resolution=300.0)
    print(f"Saved PDF: {pdf_path}")


# Batch Pipeline
def batch_clean_documents(mat_file_path, input_folder, output_folder):
    model = DnCNN(channels=1, num_of_layers=17)
    load_h5_weights(mat_file_path, model)
    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            base_name = os.path.splitext(filename)[0]
            cleaned_image_path = os.path.join(output_folder, f"{base_name}_cleaned.png")
            pdf_output_path = os.path.join(output_folder, f"{base_name}.pdf")

            print(f"Processing: {filename}")
            denoised_image = denoise_with_cnn(model, file_path)
            final_image = post_process_document(denoised_image)

            cv2.imwrite(cleaned_image_path, final_image)
            save_as_pdf(cleaned_image_path, pdf_output_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


# Run the Pipeline
if __name__ == "__main__":
    mat_file_path = r"C:/Users/ccape/Downloads/document cleaning/noise2noise-pytorch/DnCNN/model\specifics\sigma=10.mat"
    input_folder = r"C:/Users/ccape/Downloads\document cleaning/noise2noise-pytorch/data/val"
    output_folder = r"C:/Users/ccape/Downloads/document cleaning/output_pdfs"
    batch_clean_documents(mat_file_path, input_folder, output_folder)

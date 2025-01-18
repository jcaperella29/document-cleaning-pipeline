
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm


# -----------------------
# 1. Define the Dataset
# -----------------------
class NoisyDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir=None, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.noisy_files = sorted(os.listdir(noisy_dir))
        if clean_dir:
            self.clean_files = sorted(os.listdir(clean_dir))
        else:
            self.clean_files = None

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = noisy_image / 255.0  # Normalize to [0, 1]

        if self.clean_files:
            clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
            clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            clean_image = clean_image / 255.0  # Normalize to [0, 1]
        else:
            clean_image = noisy_image  # For Noise2Noise, target can be noisy too

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return torch.tensor(noisy_image).unsqueeze(0), torch.tensor(clean_image).unsqueeze(0)  # Add channel dim


# -----------------------
# 2. Define the Model (U-Net)
# -----------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # Single output channel
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder(x)
        return x


# -----------------------
# 3. Training Function
# -----------------------
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for noisy_images, clean_images in tqdm(dataloader):
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    return model


# -----------------------
# 4. Denoise and Save
# -----------------------
def denoise_image(model, noisy_image_path, output_path, device):
    model.eval()
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    noisy_tensor = torch.tensor(noisy_image).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze(0).squeeze(0).cpu().numpy()

    denoised_image = (denoised_tensor * 255).astype(np.uint8)
    cv2.imwrite(output_path, denoised_image)
    print(f"Saved cleaned image to {output_path}")


# -----------------------
# 5. Main Execution
# -----------------------
if __name__ == "__main__":
    # Directories
    TRAIN_NOISY_DIR = "data/train"
    VAL_NOISY_DIR = "data/val"
    CLEAN_DIR = "data/clean"  # Optional, can be None

    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = NoisyDataset(TRAIN_NOISY_DIR, CLEAN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Loss, Optimizer
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Training the model...")
    trained_model = train_model(model, train_loader, optimizer, criterion, DEVICE, NUM_EPOCHS)

    # Save the trained model
    torch.save(trained_model.state_dict(), "noise2noise_model.pth")
    print("Model saved as noise2noise_model.pth")

    # Denoise a sample image
    TEST_IMAGE = "data/val/noisy_0.png"
    OUTPUT_IMAGE = "cleaned_image.png"
    denoise_image(trained_model, TEST_IMAGE, OUTPUT_IMAGE, DEVICE)

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


"""
XRD1D_CNN_2_SP_PF0:CNN(XRD-1D)--> phase fraction of Secondary phase
    function:
        输入 两相的 XRD， 预测 第二相的 相 比例

version:
    20250315
    20250314
        message.txt
    
authors:
    Sunny, Zhaoyang
"""
# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
fd = '/content'  # Or your data directory

# Load data
try:
    rawY_list = joblib.load(os.path.join(fd, 'SP_phaseFractionList_40K_20250215.pkl'))
    rawX = joblib.load(os.path.join(fd, 'XRD_patternList_40K_20250215.pkl'))
except FileNotFoundError as e:
    print(f"Error: File not found.  Please ensure the following files are in '{fd}':")
    print("  - SP_phaseFractionList_40K_20250215.pkl")
    print("  - XRD_patternList_40K_20250215.pkl")
    raise e

# Convert to numpy arrays
rawX = np.array(rawX, dtype=np.float32)
rawY = np.array(rawY_list, dtype=np.float32)

print("rawX shape:", rawX.shape)  # (40000, 8000)
print("rawY shape:", rawY.shape)  # (40000,)


"""
新的实验安排，固定 CNN 架构， rawY~【10-3，0.25】
	1-CNN(Raw XRD)  rawY  RMSE, R2
	2-CNN(log(Raw XRD))  rawY
	3-CNN(Normalization(Raw XRD))  rawY

	4-CNN(Raw XRD)  Batch Normalization(rawY)
	5-CNN(log(Raw XRD))  Batch Normalization(rawY)
	6-CNN(Normalization(Raw XRD))  Batch Normalization(rawY)
"""

# Split data, tr 0.8, val 0.1, te 0.1
X_train, X_temp, y_train, y_temp = train_test_split(rawX, rawY, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train set:", X_train.shape, y_train.shape)  # (32000, 8000) (32000,)
print("Val set:  ", X_val.shape, y_val.shape)  # (4000, 8000) (4000,)
print("Test set: ", X_test.shape, y_test.shape)   # (4000, 8000) (4000,)

# --- Dataset Class ---
class XRD_Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_data = self.X[idx].reshape(1, -1)
        y_data = self.Y[idx]
        return x_data, y_data

# --- DataLoader ---
batch_size = 32  # Increased batch size

train_dataset = XRD_Dataset(X_train, y_train)
val_dataset   = XRD_Dataset(X_val, y_val)
test_dataset  = XRD_Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) # Added num_workers
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)


# --- Improved CNN Model ---
@torch.compile
class CNN_1D_Improved(nn.Module):
    def __init__(self):
        super(CNN_1D_Improved, self).__init__()

        # --- Convolutional Layers with Batch Normalization and Dropout ---

        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, stride=1, padding=7)  # Wider kernel
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)  # Dropout after activation

        self.conv2 = nn.Conv1d(8, 16, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        # self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.pool3 = nn.MaxPool1d(kernel_size=2)
        # self.dropout3 = nn.Dropout(0.3)  # Slightly increased dropout
		#
        # self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        # self.bn4 = nn.BatchNorm1d(256)
        # self.pool4 = nn.MaxPool1d(kernel_size=2)
        # self.dropout4 = nn.Dropout(0.3)
		#
        # self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm1d(512)
        # self.pool5 = nn.MaxPool1d(kernel_size=2)
        # self.dropout5 = nn.Dropout(0.4)  # More dropout deeper in the network

        # --- Adaptive Pooling (Optional) ---
        # self.adaptive_pool = nn.AdaptiveMaxPool1d(1)  # Output size of 1

        # --- Fully Connected Layers ---
        self.relu = nn.ReLU()  # Move this line HERE
        # Calculate the flattened size dynamically
        self.flattened_size = self._get_flattened_size()
        # self.fc1 = nn.Linear(512, 256)  # If using AdaptiveMaxPool1d, use this
        self.fc1 = nn.Linear(self.flattened_size, 256) # Adjust based on input size and pooling
        self.dropout_fc1 = nn.Dropout(0.5)  # Dropout in FC layer
        self.fc2 = nn.Linear(256, 1)


        # Consider LeakyReLU for potential improvement with dying ReLU
        # self.relu = nn.LeakyReLU(0.1)


    def _get_flattened_size(self):
        # Helper function to calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 8000) # Batch size 1, 1 channel, 8000 length
            dummy_output = self._conv_layers(dummy_input)
            return dummy_output.view(1, -1).size(1)


    def _conv_layers(self, x):
      #  Helper function for the conv layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        # x = self.dropout3(x)
		#
        # x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        # x = self.dropout4(x)
		#
        # x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        # x = self.dropout5(x)

        return x

    def forward(self, x):
        x = self._conv_layers(x)
        # x = self.adaptive_pool(x)  # Use if adaptive pooling is enabled

        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu(self.fc1(x))
        x = self.dropout_fc1(x) # Dropout before final layer
        x = self.fc2(x)

        return x


# --- Model, Loss, Optimizer ---
model = CNN_1D_Improved().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer
# Learning rate scheduler (Optional, but often helpful)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# --- Training Loop ---
num_epochs = 30  # Increased epochs
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1) # Reshape targets

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs_val, targets_val in val_loader:
            inputs_val = inputs_val.to(device)
            targets_val = targets_val.to(device).view(-1, 1)
            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, targets_val)
            val_loss += loss_val.item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss) # Step the scheduler

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")


# --- Evaluation ---
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs_test, targets_test in test_loader:
        inputs_test = inputs_test.to(device)
        targets_test = targets_test.to(device)  # No need to reshape here
        outputs_test = model(inputs_test)
        all_preds.extend(outputs_test.cpu().numpy().flatten()) # Use extend and flatten
        all_targets.extend(targets_test.cpu().numpy().flatten())

all_preds = np.array(all_preds)   # Convert to numpy arrays
all_targets = np.array(all_targets)


# --- Metrics and Plotting ---
r2 = r2_score(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds)) # Calculate RMSE

print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: {rmse:.6f}")

# Parity Plot
plt.figure(figsize=(6, 6))
plt.scatter(all_targets, all_preds, s=5, alpha=0.5)
plt.plot([0, 0.25], [0, 0.25], 'r--')
plt.xlabel("True SP_phaseFraction")
plt.ylabel("Predicted SP_phaseFraction")
plt.title(f"Parity Plot (R²={r2:.3f}, RMSE={rmse:.4f})")
plt.xlim([0, 0.25])
plt.ylim([0, 0.25])
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- Save the Model ---
torch.save(model.state_dict(), 'cnn_model.pth')
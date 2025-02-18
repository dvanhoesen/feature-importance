import os, sys
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Functions
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Custom 
import models
import functions as func


# Example usage
if __name__ == "__main__":

    flag_plot = True

    # Check if MPS (Mac Metal Performance Shaders) is available MacOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Check if CUDA (NVIDIA GPU) is available
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # Default to CPU
    else:
        device = torch.device("cpu")

    print(f"\nDevice: {device}")

    basepath = os.path.join("data", "california_housing")
    file_names = os.path.join(basepath, "x_names.npy")
    file_x = os.path.join(basepath, "x.npy")
    file_y = os.path.join(basepath, "y_MedHouseValue.npy")

    x = np.load(file_x)
    y = np.load(file_y)
    names = np.load(file_names)

    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    print("names shape: ", names.shape)
    print(names)

    # Normalize between 0 and 1
    max_house_value = np.amax(y)
    y = y / max_house_value
    x = x / x.max(axis=1, keepdims=True)

    print("Max Median Home Value: ${}".format(int(max_house_value * 100000)))


    N = x.shape[1]  # Number of input features
    print("Number of input features: {}\n".format(N))
    
    hidden_size = 16
    epochs = 10
    
    
    # Convert to torch tensors
    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Split into training and testing sets
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    
    model = models.SimpleNN(input_size=N, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.01)
    
    # Train the model
    func.train_model(model, train_loader, criterion, optimizer, device, num_epochs=epochs)
    
    # Test the model
    predictions, actuals = func.test_model(model, test_loader, device)

    predictions = max_house_value * 100000 * predictions # prior to calculation in $100,000
    actuals = max_house_value * 100000 * actuals # prior to calculation in $100,000
    
    if flag_plot:
        fig, ax = plt.subplots(figsize=(10,7))
        ax.scatter(actuals, predictions, color='black')
        ax.set_xlabel("Actual Median Household Value ($)", fontsize=12)
        ax.set_ylabel("Predicted Value ($)", fontsize=12)
        plt.show()
import os, sys
import numpy as np

# PyTorch Functions
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Custom 
import models as models
import functions as func


# Example usage
if __name__ == "__main__":

    # Check if MPS (Mac Metal Performance Shaders) is available MacOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Check if CUDA (NVIDIA GPU) is available
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # Default to CPU
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")



    basepath = os.path.join("data", "california_housing")
    print("Basepath: ", basepath)

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
    y = y / np.amax(y)
    x = x / x.max(axis=1, keepdims=True)


    N = x.shape[1]  # Number of input features
    print("Number of input features: ", N)
    
    hidden_size = 16
    
    
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
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Train the model
    func.train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    
    # Test the model
    predictions, actuals = func.test_model(model, test_loader, device)
    
    # Evaluate new data
    new_data = torch.randn(5, N)
    results = func.evaluate_data(model, new_data, device)
    print("Evaluated Results:", results)




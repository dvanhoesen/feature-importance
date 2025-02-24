import os, sys
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Functions
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Custom 
import models
import functions as func


def build_model(num_hidden_layers, nodes_per_layer, activation='relu', input_dim=1, output_dim=1):

    # If a single integer is provided, replicate it for all hidden layers.
    if isinstance(nodes_per_layer, int):
        nodes_per_layer = [nodes_per_layer] * num_hidden_layers
    elif isinstance(nodes_per_layer, list):
        if len(nodes_per_layer) != num_hidden_layers:
            raise ValueError("Length of nodes_per_layer list must equal num_hidden_layers")
    else:
        raise TypeError("nodes_per_layer must be an int or list of ints")
    
    # Map activation string to actual activation function.
    activation = activation.lower()
    if activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'sigmoid':
        act_fn = nn.Sigmoid()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    else:
        raise ValueError("Unsupported activation function. Choose 'relu', 'sigmoid', or 'tanh'.")
    
    layers = []
    current_dim = input_dim
    
    # Build hidden layers.
    for nodes in nodes_per_layer:
        layers.append(nn.Linear(current_dim, nodes))
        layers.append(act_fn)
        current_dim = nodes
    
    # Final output layer.
    layers.append(nn.Linear(current_dim, output_dim))
    
    return nn.Sequential(*layers)

def func_sine(x):
    return np.sin(2 * np.pi * x)
def func_power(x, n):
    return np.power(x, n)
def func_exp(x):
    return np.exp(x)
def func_log(x):
    return np.log(x)
def func_1x(x):
    return 1/x

class FunctionDataset(Dataset):
    def __init__(self, num_samples, function='sine', noise_std=0.05):
        self.num_samples = num_samples
        self.noise_std = noise_std
        
        # Define the function; sine is normalized to [0,1]
        if function == 'sine':
            self.func = func_sine
        elif function == 'quadratic':
            self.func = lambda x: func_power(x, 2)
        elif function == 'cubic':
            self.func = lambda x: func_power(x, 3)
        elif function == 'exponential':
            self.func = func_exp
        elif function == 'log':
            self.func = func_log
        elif function == '1/x':
            self.func = func_1x
        else:
            raise ValueError("Unknown function. Choose 'sine' or 'quadratic'.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random input in [0, 1]
        x = np.random.rand()
        y = self.func(x) + np.random.randn() * self.noise_std
        return torch.tensor([x], dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


if __name__ == "__main__":

    funcs = ['sine', 'quadratic', 'cubic', 'exponential', 'log', '1/x']
    activation_functions = ['relu', 'sigmoid', 'tanh']
    number_hidden_layers = range(1,11)
    number_nodes_per_layer = range(1,9)

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

    

    ### Loop over the functions, activation functions, hidden layers, and nodes per layer
    func = funcs[0]
    activation_func = activation_functions[1]
    hidden_layers = number_hidden_layers[2]
    num_nodes = 64 #number_nodes_per_layer[8]


    # Create the dataset and dataloader
    dataset = FunctionDataset(num_samples=1000, function=func, noise_std=0.05)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


    epochs = 50
    criterion = nn.MSELoss()
    model = build_model(num_hidden_layers=hidden_layers, nodes_per_layer=num_nodes, activation=activation_func)
    optimizer = optim.NAdam(model.parameters(), lr=0.01)


    for epoch in range(epochs):
        for x_batch, y_batch, in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


    x_range = np.linspace(0.001, 1, 1000)
    x_tensor = torch.tensor(x_range, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        y_predict = model(x_tensor).squeeze().numpy()

    if func == 'sine':
        y_function = func_sine(x_range)
    elif func == 'quadratic':
        y_function = func_power(x_range, 2)
    elif func == 'cubic':
        y_function = func_power(x_range, 3)
    elif func == 'exponential':
        y_function = func_exp(x_range)
    elif func == 'log':
        y_function = func_log(x_range)
    elif func == '1/x':
        y_function = func_1x(x_range)

    
    if flag_plot:
        fig, ax = plt.subplots(figsize=(10,7))
        ax.scatter(x_range, y_predict, color='black', label='predicted')
        ax.plot(x_range, y_function, color='red', label='function')
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.legend(loc='best')
        plt.show()
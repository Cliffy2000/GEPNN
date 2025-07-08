import torch
import torch.nn.functional as F

def sigmoid_fn(x):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    return torch.sigmoid(x)

def tanh_fn(x):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    return torch.tanh(x)

def relu_fn(x):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    return F.relu(x)

def linear_fn(x):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    return x

def add_fn(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x + y

def multiply_fn(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x * y

def subtract_fn(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x - y

def get_functions():
    return [
        (sigmoid_fn, 1),
        (tanh_fn, 1),
        (relu_fn, 1),
        # (linear_fn, 1),
        (add_fn, 2),
        (multiply_fn, 2),
        (subtract_fn, 2),
    ]
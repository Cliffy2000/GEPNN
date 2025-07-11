import torch
import torch.nn.functional as F

def add(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x + y

def add3(x, y, z):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    if isinstance(z, (int, float)):
        z = torch.tensor(float(z))
    return x + y + z

def subtract(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x - y

def multiply(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return x * y

def relu(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return F.relu(x + y)

def relu3(x, y, z):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    if isinstance(z, (int, float)):
        z = torch.tensor(float(z))
    return F.relu(x + y + z)

def tanh(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return torch.tanh(x + y)

def tanh3(x, y, z):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    if isinstance(z, (int, float)):
        z = torch.tensor(float(z))
    return torch.tanh(x + y + z)

def sigmoid(x, y):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    return torch.sigmoid(x + y)

def sigmoid3(x, y, z):
    if isinstance(x, (int, float)):
        x = torch.tensor(float(x))
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y))
    if isinstance(z, (int, float)):
        z = torch.tensor(float(z))
    return torch.sigmoid(x + y + z)

def get_functions():
    return [
        (add, 2),
        (add3, 3),
        (subtract, 2),
        (multiply, 2),
        (relu, 2),
        (relu3, 3),
        (tanh, 2),
        (tanh3, 3),
        (sigmoid, 2),
        (sigmoid3, 3),
    ]
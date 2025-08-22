import torch
import torch.nn.functional as F

def add(x, y):
    return x + y

def add3(x, y, z):
    return x + y + z

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def relu(x, y):
    return F.relu(x + y)

def relu3(x, y, z):
    return F.relu(x + y + z)

def tanh(x, y):
    return torch.tanh(x + y)

def tanh3(x, y, z):
    return torch.tanh(x + y + z)

def sigmoid(x, y):
    return torch.sigmoid(x + y)

def sigmoid3(x, y, z):
    return torch.sigmoid(x + y + z)

def get_functions():
    return [
        (not_f, 1),
        (and_f, 2),
        (or_f, 2),
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


def not_f(x):
    return 1 - x

def and_f(x, y):
    return torch.logical_and(x > 0.5, y > 0.5).float()

def or_f(x, y):
    return torch.logical_or(x > 0.5, y > 0.5).float()

def get_functions_xor():
    return [
        (not_f, 1),
        (and_f, 2),
        (or_f, 2),
        (relu, 2),
        (tanh, 2),
        (sigmoid, 2)
    ]


import torch
import torch.nn.functional as F

# Linear aggregation functions
def add(x, y):
    return x + y

def add3(x, y, z):
    return x + y + z

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

# Unary activation functions (NEW)
def sigmoid1(x):
    return torch.sigmoid(x)

def tanh1(x):
    return torch.tanh(x)

def relu1(x):
    return F.relu(x)

# Binary activation functions (combining aggregation + activation)
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

# Logic-based functions
def not_f(x):
    return 1 - x

# UPDATED: Smooth versions of logical functions
def and_f(x, y):
    # Soft AND using sigmoid approximation
    # High output when both inputs are high
    return torch.sigmoid(10 * (x + y - 1.5))

def or_f(x, y):
    # Soft OR using sigmoid approximation
    # High output when at least one input is high
    return torch.sigmoid(10 * (x + y - 0.5))


def get_functions():
    """Returns all available functions for general use"""
    return [
        # Unary functions
        (not_f, 1),
        (sigmoid1, 1),
        (tanh1, 1),
        (relu1, 1),
        
        # Binary functions
        (and_f, 2),
        (or_f, 2),
        (add, 2),
        (subtract, 2),
        (multiply, 2),
        (relu, 2),
        (tanh, 2),
        (sigmoid, 2),
        
        # Ternary functions
        (add3, 3),
        (relu3, 3),
        (tanh3, 3),
        (sigmoid3, 3),
    ]

def get_functions_xor():
    """Returns optimized function set for XOR problem"""
    return [
        # Unary activations
        (sigmoid1, 1),
        (tanh1, 1),
        (not_f, 1),
        
        # Binary operations
        (add, 2),
        (subtract, 2),
        (multiply, 2),
        (and_f, 2),
        (or_f, 2),
        
        # Binary activations (kept for compatibility)
        (sigmoid, 2),
        (tanh, 2),
    ]
import numpy as np

CLIP_MIN = -1e6
CLIP_MAX = 1e6

# Unary functions
def not_f(x):
    return 1 - x

import warnings

def sigmoid1(x):
    x = np.clip(x, -80, 80)
    return 1 / (1 + np.exp(-x))

def tanh1(x):
    return np.tanh(x)  # Already handles large inputs

def relu1(x):
    return np.clip(np.maximum(0, x), 0, CLIP_MAX)

# Binary functions
def and_f(x, y):
    return np.clip(x * y, CLIP_MIN, CLIP_MAX)

def or_f(x, y):
    return np.clip(x + y - x * y, CLIP_MIN, CLIP_MAX)

def add(x, y):
    return np.clip(x + y, CLIP_MIN, CLIP_MAX)

def subtract(x, y):
    return np.clip(x - y, CLIP_MIN, CLIP_MAX)

def multiply(x, y):
    return np.clip(x * y, CLIP_MIN, CLIP_MAX)

def relu(x, y):
    return np.clip(np.maximum(0, x + y), 0, CLIP_MAX)

def tanh(x, y):
    return np.tanh(x + y)

def sigmoid(x, y):
    z = np.clip(x + y, -80, 80)
    return 1 / (1 + np.exp(-z))

# Ternary functions
def add3(x, y, z):
    return np.clip(x + y + z, CLIP_MIN, CLIP_MAX)

def relu3(x, y, z):
    return np.clip(np.maximum(0, x + y + z), 0, CLIP_MAX)

def tanh3(x, y, z):
    return np.tanh(x + y + z)

def sigmoid3(x, y, z):
    s = np.clip(x + y + z, -80, 80)
    return 1 / (1 + np.exp(-s))


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

        # Binary activations
        (sigmoid, 2),
        (tanh, 2),
    ]
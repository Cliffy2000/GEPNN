import numpy as np


# Linear aggregation functions
def add(x, y):
    return x + y


def add3(x, y, z):
    return x + y + z


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


# Unary activation functions
def sigmoid1(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def tanh1(x):
    return np.tanh(x)


def relu1(x):
    return np.maximum(0, x)


# Binary activation functions (combining aggregation + activation)
def relu(x, y):
    return np.maximum(0, x + y)


def relu3(x, y, z):
    return np.maximum(0, x + y + z)


def tanh(x, y):
    return np.tanh(x + y)


def tanh3(x, y, z):
    return np.tanh(x + y + z)


def sigmoid(x, y):
    sum_val = np.clip(x + y, -50, 50)
    return 1 / (1 + np.exp(-sum_val))

def sigmoid3(x, y, z):
    sum_val = np.clip(x + y + z, -50, 50)
    return 1 / (1 + np.exp(-sum_val))


# Logic-based functions
def not_f(x):
    return 1 - x


# Smooth versions of logical functions
def and_f(x, y):
    val = np.clip(10 * (x + y - 1.5), -50, 50)
    return 1 / (1 + np.exp(-val))

def or_f(x, y):
    val = np.clip(10 * (x + y - 0.5), -50, 50)
    return 1 / (1 + np.exp(-val))


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
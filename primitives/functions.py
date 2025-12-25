import numpy as np

CLIP_MIN = -1e4
CLIP_MAX = 1e4


# Unary functions
def not_f(x):
    return 1 - x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.minimum(np.maximum(0, x), CLIP_MAX)


# Binary functions
def add(x, y):
    return np.clip(x + y, CLIP_MIN, CLIP_MAX)

def subtract(x, y):
    return np.clip(x - y, CLIP_MIN, CLIP_MAX)

def multiply(x, y):
    return np.clip(x * y, CLIP_MIN, CLIP_MAX)

def and_f(x, y):
    return np.clip(x * y, CLIP_MIN, CLIP_MAX)

def or_f(x, y):
    return np.clip(x + y - x * y, CLIP_MIN, CLIP_MAX)

def relu2(x, y):
    return np.minimum(np.maximum(0, x + y), CLIP_MAX)

def tanh2(x, y):
    return np.tanh(x + y)

def sigmoid2(x, y):
    return 1 / (1 + np.exp(-np.clip(x + y, -80, 80)))


# Function sets
def get_functions():
    return [
        (not_f, 1),
        (sigmoid, 1),
        (tanh, 1),
        (relu, 1),

        (add, 2),
        (subtract, 2),
        (multiply, 2),
        (relu2, 2),
        (tanh2, 2),
        (sigmoid2, 2),
    ]

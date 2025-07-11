import torch

def get_input_names(num_inputs):
    return [f'x{i}' for i in range(num_inputs)]

def get_weight_terminal():
    return lambda: torch.randn(1).item()

def get_bias_terminal():
    return lambda: torch.randn(1).item()
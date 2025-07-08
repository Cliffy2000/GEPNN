import torch
import geppy as gep


def get_input_names(num_inputs):
    return [f'x{i}' for i in range(num_inputs)]

def get_ephemeral_terminals():
    return [
        ('bias', lambda: torch.randn(1).item()),
        ('weight', lambda: torch.randn(1).item()),
    ]
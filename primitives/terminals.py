import numpy as np


class InputTerminal:
    def __init__(self, index):
        self.index = index
        self.name = f"x{index}"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class IndexTerminal:
    def __init__(self, index):
        self.index = index
        self.name = f"@{index}"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


def get_input_terminals(num_inputs):
    return [InputTerminal(i) for i in range(num_inputs)]


def get_index_terminals(head_length):
    return [IndexTerminal(i) for i in range(head_length)]


def get_random_weight():
    return np.random.randn()


def get_random_bias():
    return np.random.randn()
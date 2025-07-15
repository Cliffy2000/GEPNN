class Node:
    def __init__(self, symbol, weight=None, bias=None, position=None):
        self.symbol     = symbol
        self.weight     = weight
        self.bias       = bias
        self.children   = []
        self.prev_value = 0.0  
        self.value      = None
        self.position   = position

    def add_child(self, node):
        self.children.append(node)

    def reset(self):
        self.value = None

    def update_prev(self):
        if self.value is not None:
            self.prev_value = self.value
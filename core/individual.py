import random
from primitives.functions import get_functions
from primitives.terminals import get_input_terminals, get_index_terminals, get_random_weight, get_random_bias

class Individual:
    def __init__(self, head_length, num_inputs, num_weights, num_biases):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases
        funcs = [func for func,_ in get_functions()]
        inputs = get_input_terminals(num_inputs)
        indices = get_index_terminals(head_length)
        max_arity = max(arity for _,arity in get_functions())
        tail_length = head_length * (max_arity - 1) + 1
        pool = funcs + inputs + indices
        self.head = [random.choice(pool) for _ in range(head_length)]
        self.tail = [random.choice(inputs) for _ in range(tail_length)]
        self.weights = [get_random_weight() for _ in range(num_weights)]
        self.biases = [get_random_bias() for _ in range(num_biases)]
        self.expression = self.head + self.tail
        self.gene = self.expression + self.weights + self.biases

    @classmethod
    def create_random(cls, head_length, num_inputs, num_weights, num_biases):
        return cls(head_length, num_inputs, num_weights, num_biases)

    def get_head(self):
        return list(self.head)

    def get_tail(self):
        return list(self.tail)

    def get_expression(self):
        return list(self.expression)

    def __len__(self):
        return len(self.gene)

    def __str__(self):
        def name(sym):
            if hasattr(sym, 'name'):
                return sym.name
            if hasattr(sym, '__name__'):
                return sym.__name__
            return str(sym)
        head_names = [name(s) for s in self.head]
        tail_names = [name(s) for s in self.tail]
        return f"Individual[Head:{head_names}|Tail:{tail_names}|Weights:{self.weights}|Biases:{self.biases}]"

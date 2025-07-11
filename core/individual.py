import geppy as gep
from primitives.functions import get_functions
from primitives.terminals import get_input_names, get_weight_terminal, get_bias_terminal


class GepnnIndividual:
    def __init__(self, head_length, num_inputs, num_weights, num_biases, weights=None, biases=None):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases

        self.primitive_set = self._create_primitive_set()
        self.gene = gep.Gene(pset=self.primitive_set, head_length=self.head_length)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [get_weight_terminal()() for _ in range(num_weights)]

        if biases is not None:
            self.biases = biases
        else:
            self.biases = [get_bias_terminal()() for _ in range(num_biases)]

    def _create_primitive_set(self):
        input_names = get_input_names(self.num_inputs)
        pset = gep.PrimitiveSet('neural_primitives', input_names=input_names)

        functions = get_functions()
        for func, arity in functions:
            pset.add_function(func, arity)

        return pset

    @classmethod
    def create_random(cls, head_length, num_inputs, num_weights, num_biases):
        return cls(head_length, num_inputs, num_weights, num_biases)

    def get_expression(self):
        return list(self.gene)

    def get_head(self):
        return list(self.gene)[:self.head_length]

    def get_tail(self):
        return list(self.gene)[self.head_length:]

    def __str__(self):
        head = [getattr(item, 'name', str(item)) for item in self.get_head()]
        tail = [getattr(item, 'name', str(item)) for item in self.get_tail()]

        return f"Individual[Head:{head}|Tail:{tail}|Weights:{self.weights}|Biases:{self.biases}]"

    def __len__(self):
        return len(self.gene) + len(self.weights) + len(self.biases)
import geppy as gep
import random
from primitives.functions import get_functions
from primitives.terminals import get_input_names, get_weight_terminal, get_bias_terminal


class GepnnIndividual:
    def __init__(self, head_length, num_inputs, num_weights, num_biases, num_connections=0,
                 weights=None, biases=None, connections=None):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases
        self.num_connections = num_connections

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

        if connections is not None:
            self.connections = connections
        else:
            self.connections = self._generate_random_connections()

    def _create_primitive_set(self):
        input_names = get_input_names(self.num_inputs)
        pset = gep.PrimitiveSet('neural_primitives', input_names=input_names)

        functions = get_functions()
        for func, arity in functions:
            pset.add_function(func, arity)

        return pset

    def _generate_random_connections(self):
        """Generate random connections array with pairs of indices or -1 for unused."""
        connections = []
        for _ in range(self.num_connections * 2):  # *2 because each connection is a pair
            # Random index from -1 to head_length-1 (equal probability)
            connections.append(random.randint(-1, self.head_length - 1))
        return connections

    @classmethod
    def create_random(cls, head_length, num_inputs, num_weights, num_biases, num_connections=0):
        return cls(head_length, num_inputs, num_weights, num_biases, num_connections)

    def get_expression(self):
        return list(self.gene)

    def get_head(self):
        return list(self.gene)[:self.head_length]

    def get_tail(self):
        return list(self.gene)[self.head_length:]

    def get_connections_pairs(self):
        """Return connections as pairs for easier viewing."""
        pairs = []
        for i in range(0, len(self.connections), 2):
            if i + 1 < len(self.connections):
                pairs.append((self.connections[i], self.connections[i + 1]))
        return pairs

    def __str__(self):
        head = [getattr(item, 'name', str(item)) for item in self.get_head()]
        tail = [getattr(item, 'name', str(item)) for item in self.get_tail()]
        conn_pairs = self.get_connections_pairs()

        return f"Individual[Head:{head}|Tail:{tail}|Weights:{self.weights}|Biases:{self.biases}|Connections:{conn_pairs}]"

    def __len__(self):
        return len(self.gene) + len(self.weights) + len(self.biases) + len(self.connections)
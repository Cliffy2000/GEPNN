import random
from primitives.functions import get_functions
from primitives.terminals import get_input_terminals, get_index_terminals, get_random_weight, get_random_bias


class Individual:
    def __init__(self, head_length, num_inputs, num_weights, num_biases, chromosome=None):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases
        self.fitness = 0.0  # Default fitness for DEAP compatibility

        # Calculate tail length
        max_arity = max(arity for _, arity in get_functions())
        tail_length = head_length * (max_arity - 1) + 1

        if chromosome is not None:
            # Create from existing chromosome
            self._from_chromosome(chromosome, tail_length)
        else:
            # Create random individual
            self._create_random(tail_length)

    def _create_random(self, tail_length):
        """Create a random individual."""
        # Get available symbols
        funcs = [func for func, _ in get_functions()]
        inputs = get_input_terminals(self.num_inputs)
        indices = get_index_terminals(self.head_length)

        # Pool for head generation
        # TODO: Consider adjusting probabilities for different symbol types
        pool = funcs + inputs + indices

        # Generate components
        self.head = [random.choice(pool) for _ in range(self.head_length)]
        self.tail = [random.choice(inputs) for _ in range(tail_length)]
        self.weights = [get_random_weight() for _ in range(self.num_weights)]
        self.biases = [get_random_bias() for _ in range(self.num_biases)]

        self._sync_gene()

    def _from_chromosome(self, chromosome, tail_length):
        """Create individual from existing chromosome."""
        # Split chromosome into components
        expr_length = self.head_length + tail_length

        # Extract components
        expression = chromosome[:expr_length]
        self.head = expression[:self.head_length]
        self.tail = expression[self.head_length:expr_length]

        # The remaining part is weights and biases
        remaining = chromosome[expr_length:]
        self.weights = remaining[:self.num_weights] if self.num_weights > 0 else []
        self.biases = remaining[self.num_weights:self.num_weights + self.num_biases] if self.num_biases > 0 else []

        self._sync_gene()

    def _sync_gene(self):
        """Synchronize gene with component parts."""
        self.expression = self.head + self.tail
        self.gene = self.expression + self.weights + self.biases

    @classmethod
    def create_random(cls, head_length, num_inputs, num_weights, num_biases):
        """Create a random individual."""
        return cls(head_length, num_inputs, num_weights, num_biases)

    @classmethod
    def from_chromosome(cls, chromosome, head_length, num_inputs, num_weights, num_biases):
        """Create individual from existing chromosome."""
        return cls(head_length, num_inputs, num_weights, num_biases, chromosome=chromosome)

    def get_head(self):
        return list(self.head)

    def get_tail(self):
        return list(self.tail)

    def get_expression(self):
        return list(self.expression)

    def mutate_weight(self, index, value):
        """Mutate a specific weight and sync gene."""
        self.weights[index] = value
        self._sync_gene()

    def mutate_bias(self, index, value):
        """Mutate a specific bias and sync gene."""
        self.biases[index] = value
        self._sync_gene()

    def mutate_expression(self, index, symbol):
        """Mutate a specific position in expression and sync gene."""
        if index < self.head_length:
            self.head[index] = symbol
        else:
            self.tail[index - self.head_length] = symbol
        self._sync_gene()

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
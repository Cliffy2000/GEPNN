import random
from primitives.functions import get_functions, get_functions_xor
from primitives.terminals import get_input_terminals, get_index_terminals, get_random_weight, get_random_bias


class Individual:
    def __init__(self, head_length, num_inputs, num_weights, num_biases, chromosome=None):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases
        self.fitness = None

        max_arity = max(arity for _, arity in get_functions())      # NOTE: potentially move outside of init for slight optimization
        self.tail_length = head_length * (max_arity - 1) + 1

        if chromosome is not None:
            self._from_chromosome(chromosome)
        else:
            self._create_random()

    def _create_random(self):
        """Create a random individual."""
        functions = [f for f, _ in get_functions()]
        inputs = get_input_terminals(self.num_inputs)
        indices = get_index_terminals(self.head_length)

        # TODO: confirm initial probabilities
        pool = functions + inputs + indices

        self.head = [random.choice(pool) for _ in range(self.head_length)]
        self.tail = [random.choice(inputs) for _ in range(self.tail_length)]
        self.weights = [get_random_weight() for _ in range(self.num_weights)]
        self.biases = [get_random_bias() for _ in range(self.num_biases)]

    def _from_chromosome(self, chromosome):
        """Create individual from existing chromosome."""
        expr_length = self.head_length + self.tail_length

        self.head = list(chromosome[:self.head_length])
        self.tail = list(chromosome[self.head_length:expr_length])
        self.weights = list(chromosome[expr_length:expr_length + self.num_weights])
        self.biases = list(chromosome[expr_length + self.num_weights:])

    @property
    def expression(self):
        """Get the full expression (head + tail)."""
        return self.head + self.tail

    @property
    def gene(self):
        """Get the complete gene (expression + weights + biases)."""
        return self.expression + self.weights + self.biases

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

    def export(self):
        def name(sym):
            if hasattr(sym, 'name'):
                return sym.name
            if hasattr(sym, '__name__'):
                return sym.__name__
            return str(sym)
        head_names = [name(s) for s in self.head]
        tail_names = [name(s) for s in self.tail]
        return {
            "Individual": {
                "Head": str(head_names),
                "Tail": str(tail_names),
                "Weights": str(self.weights),
                "Biases": str(self.biases)
            }
        }


    def copy(self):
        """Create a deep copy of this individual, preserving fitness."""
        new_individual = Individual(
            head_length=self.head_length,
            num_inputs=self.num_inputs,
            num_weights=self.num_weights,
            num_biases=self.num_biases,
            chromosome=self.gene
        )

        new_individual.fitness = self.fitness
        return new_individual


class Individual_xor(Individual):
    def _create_random(self):
        """Create a random individual."""
        functions = [f for f, _ in get_functions_xor()]
        inputs = get_input_terminals(self.num_inputs)
        indices = get_index_terminals(self.head_length)

        new_head = []
        for i in range(self.head_length):
            symbol_type = random.choice(['function', 'input'])
            if symbol_type == 'function':
                new_head.append(random.choice(functions))
            else:
                new_head.append(random.choice(inputs))
        self.head = new_head

        self.tail = [random.choice(inputs) for _ in range(self.tail_length)]
        self.weights = [get_random_weight() for _ in range(self.num_weights)]
        self.biases = [get_random_bias() for _ in range(self.num_biases)]

import random
from primitives.functions import get_functions
from primitives.terminals import get_input_terminals, get_index_terminals, get_random_weight, get_random_bias


_MAX_ARITY = None


def _get_max_arity():
    """Cache and return max arity from function set."""
    global _MAX_ARITY
    if _MAX_ARITY is None:
        _MAX_ARITY = max(arity for _, arity in get_functions())
    return _MAX_ARITY


class Individual_v2:
    """
    Individual with Index Terminals allowed in tail.
    Follows traditional GEP: uniform probability across all symbols.
    - Head: uniform across functions + inputs + indices
    - Tail: uniform across inputs + indices (terminals only)
    """
    def __init__(self, head_length, num_inputs, num_weights, num_biases, chromosome=None):
        self.head_length = head_length
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.num_biases = num_biases
        self.fitness = None

        self.tail_length = head_length * (_get_max_arity() - 1) + 1

        if chromosome is not None:
            self._from_chromosome(chromosome)
        else:
            self._create_random()

    def _create_random(self):
        """Create a random individual with Index Terminals in tail.
        Traditional GEP: uniform probability across all valid symbols.
        """
        functions = [f for f, _ in get_functions()]
        inputs = get_input_terminals(self.num_inputs)
        indices = get_index_terminals(self.head_length)

        head_pool = functions + inputs + indices  # All symbols for head
        tail_pool = inputs + indices              # Only terminals for tail

        self.head = [random.choice(head_pool) for _ in range(self.head_length)]
        self.tail = [random.choice(tail_pool) for _ in range(self.tail_length)]
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
        return f"Individual_v2[Head:{head_names}|Tail:{tail_names}|Weights:{self.weights}|Biases:{self.biases}]"

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
        new_individual = Individual_v2(
            head_length=self.head_length,
            num_inputs=self.num_inputs,
            num_weights=self.num_weights,
            num_biases=self.num_biases,
            chromosome=self.gene
        )
        new_individual.fitness = self.fitness
        return new_individual


class Individual_v2_xor(Individual_v2):
    """
    Individual without Index Terminals for non-temporal tasks (e.g., XOR).
    Inherits from Individual_v2, overrides only initialization.
    - Head: uniform across functions + inputs (no indices)
    - Tail: inputs only
    """
    def _create_random(self):
        """Create a random individual without Index Terminals."""
        functions = [f for f, _ in get_functions()]
        inputs = get_input_terminals(self.num_inputs)

        head_pool = functions + inputs  # No indices
        tail_pool = inputs              # Inputs only

        self.head = [random.choice(head_pool) for _ in range(self.head_length)]
        self.tail = [random.choice(tail_pool) for _ in range(self.tail_length)]
        self.weights = [get_random_weight() for _ in range(self.num_weights)]
        self.biases = [get_random_bias() for _ in range(self.num_biases)]

    def copy(self):
        """Create a deep copy of this individual, preserving fitness."""
        new_individual = Individual_v2_xor(
            head_length=self.head_length,
            num_inputs=self.num_inputs,
            num_weights=self.num_weights,
            num_biases=self.num_biases,
            chromosome=self.gene
        )
        new_individual.fitness = self.fitness
        return new_individual
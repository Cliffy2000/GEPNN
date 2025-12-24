import random
from primitives.functions import get_functions
from primitives.terminals import get_input_terminals, get_index_terminals


_cache_temporal = {}
_cache_non_temporal = {}


def _get_symbols_temporal(head_length, num_inputs):
    """Cache and return symbol pools for temporal tasks (with Index Terminals)."""
    key = (head_length, num_inputs)

    if key not in _cache_temporal:
        functions = [f for f, _ in get_functions()]
        indices = get_index_terminals(head_length)
        inputs = get_input_terminals(num_inputs)

        head_pool = functions + inputs + indices  # Functions + terminals
        tail_pool = inputs + indices  # Terminals (inputs + indices)

        _cache_temporal[key] = (head_pool, tail_pool)

    return _cache_temporal[key]


def _get_symbols_non_temporal(head_length, num_inputs):
    """Cache and return symbol pools for non-temporal tasks (no Index Terminals)."""
    key = (head_length, num_inputs)

    if key not in _cache_non_temporal:
        functions = [f for f, _ in get_functions()]
        inputs = get_input_terminals(num_inputs)

        head_pool = functions + inputs  # Functions + inputs
        tail_pool = inputs  # Inputs only

        _cache_non_temporal[key] = (head_pool, tail_pool)

    return _cache_non_temporal[key]


def crossover_one_point(indv1, indv2, crossover_rate):
    """
    Performs one point crossover on two parents at the crossover_rate.
    Nothing changes if the probability is not met.

    :param indv1: parent 1
    :param indv2: parent 2
    :param crossover_rate: the probability that a crossover is performed
    :return: the genes of offspring1 and offspring2
    """
    gene1 = indv1.gene
    gene2 = indv2.gene

    crossover_point = random.randint(1, len(indv1.gene) - 1)

    if random.random() < crossover_rate:
        new_gene1 = gene1[:crossover_point] + gene2[crossover_point:]
        new_gene2 = gene2[:crossover_point] + gene1[crossover_point:]
        return new_gene1, new_gene2

    return gene1, gene2


def mutate_v2(indv, mutation_rate):
    """
    Mutation for temporal tasks with Index Terminals in head and tail.

    Follows traditional GEP (Ferreira 2001):
    - Head: uniform probability across functions + terminals (inputs + indices)
    - Tail: uniform probability across terminals (inputs + indices)
    - Coefficients: Gaussian perturbation

    :param indv: the original individual
    :param mutation_rate: probability that a single allele is mutated
    :return: the mutated gene
    """
    head_pool, tail_pool = _get_symbols_temporal(indv.head_length, indv.num_inputs)

    # Head mutation: functions or terminals
    new_head = []
    for i in range(indv.head_length):
        if random.random() < mutation_rate:
            new_head.append(random.choice(head_pool))
        else:
            new_head.append(indv.gene[i])

    # Tail mutation: terminals (inputs + indices)
    new_tail = []
    for i in range(indv.tail_length):
        if random.random() < mutation_rate:
            new_tail.append(random.choice(tail_pool))
        else:
            new_tail.append(indv.gene[indv.head_length + i])

    # Coefficient mutation: Gaussian perturbation
    new_coeff = []
    for i in range(indv.num_weights + indv.num_biases):
        if random.random() < mutation_rate:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i] + random.gauss(0, 1))
        else:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i])

    return new_head + new_tail + new_coeff


def mutate_v2_xor(indv, mutation_rate):
    """
    Mutation for non-temporal tasks without Index Terminals.

    For non-temporal tasks (e.g., XOR):
    - Head: uniform probability across functions + inputs
    - Tail: inputs only
    - Coefficients: Gaussian perturbation

    :param indv: the original individual
    :param mutation_rate: probability that a single allele is mutated
    :return: the mutated gene
    """
    head_pool, tail_pool = _get_symbols_non_temporal(indv.head_length, indv.num_inputs)

    # Head mutation: functions or inputs
    new_head = []
    for i in range(indv.head_length):
        if random.random() < mutation_rate:
            new_head.append(random.choice(head_pool))
        else:
            new_head.append(indv.gene[i])

    # Tail mutation: inputs only
    new_tail = []
    for i in range(indv.tail_length):
        if random.random() < mutation_rate:
            new_tail.append(random.choice(tail_pool))
        else:
            new_tail.append(indv.gene[indv.head_length + i])

    # Coefficient mutation: Gaussian perturbation
    new_coeff = []
    for i in range(indv.num_weights + indv.num_biases):
        if random.random() < mutation_rate:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i] + random.gauss(0, 1))
        else:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i])

    return new_head + new_tail + new_coeff
import random
from core.individual import Individual, Individual_xor
from primitives.functions import get_functions, get_functions_xor
from primitives.terminals import get_input_terminals, get_index_terminals

_cache = {}
_cache_xor = {}

def _get_symbol(head_length, num_inputs):
    key = (head_length, num_inputs)

    if key not in _cache:
        functions = [f for f, _ in get_functions()]
        indices = get_index_terminals(head_length)
        inputs = get_input_terminals(num_inputs)
        _cache[key] = (functions, indices, inputs)

    return _cache[key]

def _get_symbol_xor(head_length, num_inputs):
    key = (head_length, num_inputs)

    if key not in _cache_xor:
        functions = [f for f, _ in get_functions_xor()]
        indices = get_index_terminals(head_length)
        inputs = get_input_terminals(num_inputs)
        _cache_xor[key] = (functions, indices, inputs)

    return _cache_xor[key]


def crossover_one_point(indv1, indv2, crossover_rate):
    """
    Performs one point crossover on two parents at the crossover_rate, nothing changes if the probability is not met.
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


def mutate(indv: Individual, mutation_rate):
    """
    Perform allele-wise mutation according to the mutation_rate on the gene of the given individual.
    :param indv: the original individual
    :param mutation_rate: the probability that a single allele is mutated
    :return: the mutated gene
    """
    # NOTE: Room for optimization with numpy
    functions, indices, inputs = _get_symbol(indv.head_length, indv.num_inputs)

    new_head = []
    for i in range(indv.head_length):
        if random.random() < mutation_rate:
            symbol_type = random.choice(['function', 'index', 'input'])
            if symbol_type == 'function':
                new_head.append(random.choice(functions))
            elif symbol_type == 'index':
                new_head.append(random.choice(indices))
            else:
                new_head.append(random.choice(inputs))
        else:
            # this allele in the head is not mutated
            new_head.append(indv.gene[i])

    new_tail = []
    for i in range(indv.tail_length):
        if random.random() < mutation_rate:
            new_tail.append(random.choice(inputs))
        else:
            new_tail.append(indv.gene[indv.head_length + i])

    new_coeff = []
    for i in range(indv.num_weights + indv.num_biases):
        if random.random() < mutation_rate:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i] + random.gauss(0, 1))
        else:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i])

    new_gene = new_head + new_tail + new_coeff
    return new_gene


def mutate_xor(indv: Individual_xor, mutation_rate):
    """
    Perform allele-wise mutation according to the mutation_rate on the gene of the given individual.
    :param indv: the original individual
    :param mutation_rate: the probability that a single allele is mutated
    :return: the mutated gene
    """
    # NOTE: Room for optimization with numpy
    functions, indices, inputs = _get_symbol_xor(indv.head_length, indv.num_inputs)

    new_head = []
    for i in range(indv.head_length):
        if random.random() < mutation_rate:
            symbol_type = random.choice(['function', 'index', 'input'])
            if symbol_type == 'function':
                new_head.append(random.choice(functions))
            elif symbol_type == 'index':
                new_head.append(random.choice(indices))
            else:
                new_head.append(random.choice(inputs))
        else:
            # this allele in the head is not mutated
            new_head.append(indv.gene[i])

    new_tail = []
    for i in range(indv.tail_length):
        if random.random() < mutation_rate:
            new_tail.append(random.choice(inputs))
        else:
            new_tail.append(indv.gene[indv.head_length + i])

    new_coeff = []
    for i in range(indv.num_weights + indv.num_biases):
        if random.random() < mutation_rate:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i] + random.gauss(0, 1))
        else:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i])

    new_gene = new_head + new_tail + new_coeff
    return new_gene


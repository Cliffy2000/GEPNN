import random
from core.individual import Individual
from primitives.functions import get_functions
from primitives.terminals import get_input_terminals, get_index_terminals

_cache = {}

def _get_symbols(head_length, num_inputs):
    key = (head_length, num_inputs)
    if key not in _cache:
        functions = [f for f, _ in get_functions()]
        inputs = get_input_terminals(num_inputs)
        indices = get_index_terminals(head_length)     # NOTE: potentially change to head_length + tail_length
        _cache[key] = (functions, inputs, indices)
    return _cache[key]


def crossover_one_point(indv1: Individual, indv2: Individual):
    """
    Performs crossover between two chromosome individuals
    :param indv1: first parent
    :param indv2: second parent
    :return: Two new offspring
    """
    gene1 = indv1.gene
    gene2 = indv2.gene

    crossover_point = random.randint(1, len(gene1) - 1)

    new_gene1 = gene1[:crossover_point] + gene2[crossover_point:]
    new_gene2 = gene2[:crossover_point] + gene1[crossover_point:]

    offspring1 = Individual(
        head_length = indv1.head_length,
        num_inputs = indv1.num_inputs,
        num_weights = indv1.num_weights,
        num_biases = indv1.num_biases,
        chromosome = new_gene1
    )
    offspring2 = Individual(
        head_length=indv1.head_length,
        num_inputs=indv1.num_inputs,
        num_weights=indv1.num_weights,
        num_biases=indv1.num_biases,
        chromosome=new_gene2
    )

    return offspring1, offspring2


def mutate(indv: Individual):
    """
    The mutation operator if mutation takes place, operates on the head / tail / weights & biases separately
    :param indv: Individual to mutate
    :return: New individual
    """
    functions, inputs, indices = _get_symbols(indv.head_length, indv.num_inputs)
    head_symbols = functions + inputs + indices
    tail_symbols = inputs       # NOTE: could be changed to inputs + indices

    new_gene = (
        [random.choice(head_symbols) for _ in indv.head] +
        [random.choice(tail_symbols) for _ in indv.tail] +
        [w + random.gauss(0, 1) for w in indv.weights] +
        [b + random.gauss(0, 1) for b in indv.biases]
    )

    return Individual(
        head_length = indv.head_length,
        num_inputs = indv.num_inputs,
        num_weights = indv.num_weights,
        num_biases = indv.num_biases,
        chromosome = new_gene
    )


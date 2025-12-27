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


def crossover_sync(indv1, indv2, crossover_rate):
    """
    Synchronized crossover that maintains structure-parameter pairing.

    When structural components are exchanged at a crossover point, the
    associated weights (edge weights) and biases are exchanged together.
    This preserves learned parameter associations.

    Weight indexing: weight[i] corresponds to edge leading to node i (with
    placeholder at index 0). Weights array has length = head_length + tail_length.

    Bias indexing: bias[i] corresponds to head position i. Biases array has
    length = head_length. Only biases at positions containing functions are active.

    :param indv1: parent 1
    :param indv2: parent 2
    :param crossover_rate: the probability that a crossover is performed
    :return: the genes of offspring1 and offspring2
    """
    if random.random() >= crossover_rate:
        return indv1.gene, indv2.gene

    # Structure region: head + tail
    struct_length = indv1.head_length + indv1.tail_length

    # Choose crossover point in structure (at least 1 from each end)
    struct_point = random.randint(1, struct_length - 1)

    # Extract structures
    struct1 = indv1.gene[:struct_length]
    struct2 = indv2.gene[:struct_length]

    # Create new structures
    new_struct1 = struct1[:struct_point] + struct2[struct_point:]
    new_struct2 = struct2[:struct_point] + struct1[struct_point:]

    # Weights: weight[i] corresponds to node i+1 (node 0 has no incoming edge)
    # So crossover at struct_point means swap weights from index struct_point-1 onward
    weight_point = max(0, struct_point - 1)

    weights1 = list(indv1.weights)
    weights2 = list(indv2.weights)

    new_weights1 = weights1[:weight_point] + weights2[weight_point:]
    new_weights2 = weights2[:weight_point] + weights1[weight_point:]

    # Biases: length = head_length, indexed by head position
    # Crossover at same point, but capped to head_length
    bias_point = min(struct_point, indv1.head_length)

    biases1 = list(indv1.biases)
    biases2 = list(indv2.biases)

    new_biases1 = biases1[:bias_point] + biases2[bias_point:]
    new_biases2 = biases2[:bias_point] + biases1[bias_point:]

    # Combine into full genes
    new_gene1 = new_struct1 + new_weights1 + new_biases1
    new_gene2 = new_struct2 + new_weights2 + new_biases2

    return new_gene1, new_gene2


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


def mutate_v2_reg(indv, mutation_rate):
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
            base_sigma = 10 ** random.uniform(-2, 0)  # [0.01, 1.0]
            old_val = indv.gene[indv.head_length + indv.tail_length + i]
            if random.random() < 0.5:
                sigma = base_sigma  # Fine-tuning: always small
            else:
                sigma = base_sigma * max(1.0, abs(old_val))  # Scaled by magnitude

            new_val = old_val + random.gauss(0, sigma)
            new_coeff.append(new_val)
        else:
            new_coeff.append(indv.gene[indv.head_length + indv.tail_length + i])

    return new_head + new_tail + new_coeff

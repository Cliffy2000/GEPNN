import torch
import random
import numpy as np
from deap import base, creator, tools, algorithms
from core.individual import Individual_xor
from core.network import Network
from core.operators import crossover_one_point, mutate_xor
from utils.output_utils import print_header, print_section_break, format_list
from evaluation.fitness import evaluate_xor

# GA Parameters
POPULATION_SIZE = 400
N_GENERATIONS = 2500
CXPB = 0.7  # Crossover probability
MUTPB = 0.15  # Mutation probability
TOURNSIZE = 25  # Tournament selection size

# GEP Parameters for XOR
HEAD_LENGTH = 10  # Smaller head for simpler XOR problem
NUM_INPUTS = 2  # XOR has 2 inputs
NUM_WEIGHTS = 20  # Fewer weights needed
NUM_BIASES = 10  # Fewer biases needed

# Create DEAP types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("GEPIndividual", Individual_xor, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()


def create_individual():
    """Create a GEP individual with XOR-specific initialization."""
    ind = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES
    )
    # Initialize the fitness attribute
    ind.fitness = creator.FitnessMax()
    return ind


def crossover_wrapper(ind1, ind2):
    """Wrapper for crossover that maintains DEAP compatibility."""
    # Perform crossover
    offspring1, offspring2 = crossover_one_point(ind1, ind2)

    # Create new DEAP individuals from offspring
    new_ind1 = creator.GEPIndividual(
        head_length=offspring1.head_length,
        num_inputs=offspring1.num_inputs,
        num_weights=offspring1.num_weights,
        num_biases=offspring1.num_biases,
        chromosome=offspring1.gene
    )
    new_ind2 = creator.GEPIndividual(
        head_length=offspring2.head_length,
        num_inputs=offspring2.num_inputs,
        num_weights=offspring2.num_weights,
        num_biases=offspring2.num_biases,
        chromosome=offspring2.gene
    )

    # Initialize fitness attributes
    new_ind1.fitness = creator.FitnessMax()
    new_ind2.fitness = creator.FitnessMax()

    return new_ind1, new_ind2


def mutation_wrapper(ind):
    """Wrapper for mutation that maintains DEAP compatibility."""
    # Perform mutation
    mutated = mutate_xor(ind)

    # Create new DEAP individual from mutated
    new_ind = creator.GEPIndividual(
        head_length=mutated.head_length,
        num_inputs=mutated.num_inputs,
        num_weights=mutated.num_weights,
        num_biases=mutated.num_biases,
        chromosome=mutated.gene
    )

    # Initialize fitness attribute
    new_ind.fitness = creator.FitnessMax()

    return (new_ind,)  # Must return a tuple containing the individual


# Register functions with toolbox
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_xor)
toolbox.register("mate", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)


def test_xor_solution(network):
    """Test the network on all 4 XOR cases."""
    print("\nTesting XOR Truth Table:")
    print("Input | Expected | Actual | Correct")
    print("-" * 40)

    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]

    all_correct = True

    for inputs, expected in test_cases:
        input_dict = {'x0': float(inputs[0]), 'x1': float(inputs[1])}

        try:
            output = network.forward(input_dict)
            output_val = output.item() if torch.is_tensor(output) else float(output)
            predicted = 1 if output_val > 0.5 else 0

            correct = predicted == expected
            all_correct &= correct

            print(f"{inputs[0]}, {inputs[1]}   |    {expected}     | {output_val:.3f} ({predicted}) | {'✓' if correct else '✗'}")
        except Exception as e:
            print(f"{inputs[0]}, {inputs[1]}   |    {expected}     | Error | ✗")
            all_correct = False

    return all_correct


def main():
    """Main evolutionary algorithm loop for XOR problem."""
    print_header("GEP EVOLUTIONARY ALGORITHM - XOR PROBLEM", level=1)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Generations: {N_GENERATIONS}")
    print(f"Head Length: {HEAD_LENGTH}")
    print(f"Number of Inputs: {NUM_INPUTS}")
    print(f"Crossover Probability: {CXPB}")
    print(f"Mutation Probability: {MUTPB}")

    # Create initial population
    print_section_break()
    print("Creating initial population...")
    population = toolbox.population(n=POPULATION_SIZE)

    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame to track best individuals
    hof = tools.HallOfFame(1)

    # Evolution loop
    print_section_break()
    print("Starting evolution...")

    # Calculate elitism size (10% of population)
    elite_size = int(0.1 * POPULATION_SIZE)
    print(f"Using elitism with {elite_size} elite individuals")

    for gen in range(N_GENERATIONS):
        # Store elite individuals before creating offspring
        elite = tools.selBest(population, elite_size)

        # Select the next generation (only need to fill non-elite spots)
        offspring_size = POPULATION_SIZE - elite_size
        offspring = toolbox.select(population, offspring_size)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1_new, child2_new = toolbox.mate(child1, child2)
                offspring[offspring.index(child1)] = child1_new
                offspring[offspring.index(child2)] = child2_new

        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                mutated = toolbox.mutate(mutant)
                offspring[offspring.index(mutant)] = mutated[0]

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine elite and offspring to form new population
        population[:] = elite + offspring

        # Update hall of fame
        hof.update(population)

        # Gather and print statistics
        record = stats.compile(population)
        print(f"\nGeneration {gen + 1}:")
        print(f"  Min: {record['min']:.4f}")
        print(f"  Max: {record['max']:.4f}")
        print(f"  Avg: {record['avg']:.4f}")
        print(f"  Std: {record['std']:.4f}")
        print(f"  Best ever: {hof[0].fitness.values[0]:.4f}")

        # Check for perfect solution
        if record['max'] >= 1.0:
            print("\n*** PERFECT SOLUTION FOUND! ***")
            break

    # Print final results
    print_section_break()
    print_header("EVOLUTION COMPLETE", level=2)
    print(f"\nBest individual found:")
    best_ind = hof[0]
    print(f"  Fitness: {best_ind.fitness.values[0]:.4f}")
    print(f"\n  Expression: {best_ind}")

    # Create and display the best network
    best_network = Network(best_ind)
    print("\nBest Network Structure:")
    best_network.print_structure()

    # Test the best network on XOR truth table
    print_section_break()
    print_header("TESTING BEST NETWORK ON XOR", level=2)
    perfect = test_xor_solution(best_network)

    if perfect:
        print("\n✓ Network correctly solves XOR problem!")
    else:
        print("\n✗ Network does not perfectly solve XOR problem")

    return population, hof


if __name__ == "__main__":


    # Run the GA
    final_population, hall_of_fame = main()

    print("\n" + "=" * 80)
    print("XOR trial execution completed successfully!")
import torch
import numpy as np
from deap import base, creator, tools
from core.individual import Individual_xor
from core.network import Network
from core.operators import crossover_one_point, mutate_xor
from utils.output_utils import print_header, print_section_break
from evaluation.fitness import evaluate_xor


# GA Parameters
POPULATION_SIZE = 1000
GENERATION_COUNT = 50
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 8

# GEP Parameters
HEAD_LENGTH = 8    # tail length = 8 * (2 - 1) + 1 = 9
NUM_INPUTS = 2
NUM_WEIGHTS = 16    # all nodes apart from the root node need a weight
NUM_BIASES = 8     # all head nodes need a bias

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("GEPIndividual", Individual_xor, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def create_individual():
    indv = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES
    )

    indv.fitness = creator.FitnessMax()
    return indv


def crossover_wrapper(indv1, indv2):
    offspring1, offspring2 = crossover_one_point(indv1, indv2, CROSSOVER_RATE)

    new_indv1 = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES,
        chromosome = offspring1
    )
    new_indv2 = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES,
        chromosome = offspring2
    )

    new_indv1.fitness = creator.FitnessMax()
    new_indv2.fitness = creator.FitnessMax()

    return new_indv1, new_indv2


def mutation_wrapper(indv):
    mutated = mutate_xor(indv, MUTATION_RATE)

    new_indv = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES,
        chromosome = mutated
    )

    new_indv.fitness = creator.FitnessMax()
    return (new_indv, )     # must return a tuple that contains the individual


toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_xor)
toolbox.register("mate", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


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
    print(f"Generations: {GENERATION_COUNT}")
    print(f"Head Length: {HEAD_LENGTH}")
    print(f"Number of Inputs: {NUM_INPUTS}")
    print(f"Crossover Probability: {CROSSOVER_RATE}")
    print(f"Mutation Probability: {MUTATION_RATE}")

    # Create initial population
    print_section_break()
    print("Creating initial population...")
    population = toolbox.population(n=POPULATION_SIZE)

    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for indv, fit in zip(population, fitnesses):
        indv.fitness.values = fit

    # Statistics
    stats = tools.Statistics(lambda indv: indv.fitness.values)
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

    for gen in range(GENERATION_COUNT):
        # Store elite individuals before creating offspring
        elite = tools.selBest(population, elite_size)

        # Select the next generation (only need to fill non-elite spots)
        offspring_size = POPULATION_SIZE - elite_size
        offspring = toolbox.select(population, offspring_size)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            child1_new, child2_new = toolbox.mate(child1, child2)
            offspring[offspring.index(child1)] = child1_new
            offspring[offspring.index(child2)] = child2_new

        # Apply mutation
        # TODO: fix mutate
        for mutant in offspring:
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
    best_network.print_tree()

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

import random
import numpy as np
from deap import base, creator, tools, algorithms
from core.individual import Individual
from core.network import Network
from core.operators import crossover_one_point, mutate
from utils.output_utils import print_header, print_section_break, format_list

# GA Parameters
POPULATION_SIZE = 100
N_GENERATIONS = 50
CXPB = 0.7  # Crossover probability
MUTPB = 0.2  # Mutation probability
TOURNSIZE = 3  # Tournament selection size

# GEP Parameters
HEAD_LENGTH = 10
NUM_INPUTS = 3
NUM_WEIGHTS = 30
NUM_BIASES = 10

# Create DEAP types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("GEPIndividual", Individual, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()


def create_individual():
    """Create a GEP individual with random initialization."""
    ind = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES
    )
    # Initialize the fitness attribute
    ind.fitness = creator.FitnessMax()
    return ind


def evaluate_individual(individual):
    """
    Placeholder fitness function that evaluates an individual.
    Returns a tuple as required by DEAP.
    """
    # Create network from individual
    network = Network(individual)

    # Generate some test inputs
    test_inputs = {f'x{i}': random.random() for i in range(NUM_INPUTS)}

    try:
        # Run the network
        output = network.forward(test_inputs)

        # Placeholder: return random fitness between 0 and 1
        fitness = random.random()

        # You can use the output in your actual fitness calculation
        # For example: fitness = abs(output.item()) if torch.is_tensor(output) else abs(output)

    except Exception as e:
        # Handle any errors in network execution
        fitness = 0.0
        print(f"Error evaluating individual: {e}")

    return (fitness,)  # Must return a tuple


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
    mutated = mutate(ind)

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
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)


def main():
    """Main evolutionary algorithm loop."""
    print_header("GEP EVOLUTIONARY ALGORITHM", level=1)
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

    for gen in range(N_GENERATIONS):
        # Select the next generation
        offspring = toolbox.select(population, len(population))
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

        # Replace population
        population[:] = offspring

        # Update hall of fame
        hof.update(population)

        # Gather and print statistics
        record = stats.compile(population)
        print(f"\nGeneration {gen + 1}:")
        print(f"  Min: {record['min']:.4f}")
        print(f"  Max: {record['max']:.4f}")
        print(f"  Avg: {record['avg']:.4f}")
        print(f"  Std: {record['std']:.4f}")

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

    # Test the best network
    print_section_break()
    print_header("TESTING BEST NETWORK", level=2)
    test_inputs = {f'x{i}': random.random() for i in range(NUM_INPUTS)}
    output = best_network.forward(test_inputs)
    print(f"\nTest Input: {test_inputs}")
    print(f"Network Output: {output}")

    return population, hof


if __name__ == "__main__":
    random.seed(None)
    # Run the GA
    final_population, hall_of_fame = main()

    print("\n" + "=" * 80)
    print("Algorithm execution completed successfully!")
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import datetime
import numpy as np
from multiprocessing import Pool
from deap import base, creator, tools
from core.individual import Individual
from core.network import Network
from core.operators import crossover_one_point, mutate
from evaluation.fitness import evaluate_mnist
from utils.output_utils import print_header, print_section_break

import signal

# ==================================================

# Experiment Parameters
ITERATIONS = 1  # Run one experiment at a time for MNIST
CORES = min(25, os.cpu_count())  # Don't exceed available cores

# GA Parameters
POPULATION_SIZE = 50
MAX_GENERATION_LIMIT = 1000
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 2
ELITISM_RATIO = 0.1

# GEP Parameters
HEAD_LENGTH = 30
NUM_INPUTS = 1
NUM_WEIGHTS = 60
NUM_BIASES = 30

# ==================================================

# Global flag for interruption
interrupted = False


def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\n\nInterrupt received! Stopping gracefully...")


def init_worker():
    """Initialize worker process to ignore SIGINT"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def create_individual_wrapper():
    """
    Wrapper function for DEAP base.Toolbox
    :return: an individual
    """
    indv = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES
    )
    indv.fitness = creator.FitnessMax()
    return indv


def crossover_wrapper(indv1, indv2):
    """
    Wrapper function for DEAP base.Toolbox
    """
    offspring1, offspring2 = crossover_one_point(indv1, indv2, CROSSOVER_RATE)

    new_indv1 = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES,
        chromosome=offspring1
    )
    new_indv2 = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES,
        chromosome=offspring2
    )

    new_indv1.fitness = creator.FitnessMax()
    new_indv2.fitness = creator.FitnessMax()

    return new_indv1, new_indv2


def mutation_wrapper(indv):
    """
    Wrapper function for DEAP base.Toolbox
    """
    mutated = mutate(indv, MUTATION_RATE)

    new_indv = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES,
        chromosome=mutated
    )
    new_indv.fitness = creator.FitnessMax()
    return (new_indv,)


# Create types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("GEPIndividual", Individual, fitness=creator.FitnessMax)

# Create toolbox with parallel map
toolbox = base.Toolbox()

# CRITICAL: Create the pool here for parallel evaluation within generations
pool = Pool(processes=CORES, initializer=init_worker)
toolbox.register("map", pool.map)  # Use parallel map for fitness evaluation

toolbox.register("individual", create_individual_wrapper)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_mnist)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("crossover", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)


def run_single_ga():
    """
    Run a single GA experiment with parallel fitness evaluation
    """
    global interrupted

    print("Initializing population...")
    population = toolbox.population(n=POPULATION_SIZE)

    print(f"Evaluating initial population using {CORES} cores...")
    # This will evaluate all individuals in parallel
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for indv, fit in zip(population, fitnesses):
        indv.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    hof = tools.HallOfFame(1)
    hof.update(population)

    elite_size = int(ELITISM_RATIO * POPULATION_SIZE)
    generation = 0

    # Track statistics
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    print(f"\nStarting evolution...")
    print(f"Population: {POPULATION_SIZE}, Elite: {elite_size}")
    print("-" * 60)

    while generation < MAX_GENERATION_LIMIT and not interrupted:
        # Select elite
        elite = tools.selBest(population, elite_size)

        # Select and clone parents
        offspring_size = POPULATION_SIZE - elite_size
        parents = toolbox.select(population, offspring_size)
        parents = list(map(toolbox.clone, parents))
        offspring = []

        # Crossover
        for i in range(0, len(parents) - 1, 2):
            if interrupted:
                break
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = toolbox.crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([toolbox.clone(parents[i]), toolbox.clone(parents[i + 1])])

        if interrupted:
            break

        # Handle odd number of parents
        if len(parents) % 2:
            offspring.append(toolbox.clone(parents[-1]))

        # Mutation
        for i in range(len(offspring)):
            if np.random.random() < MUTATION_RATE:
                offspring[i] = toolbox.mutate(offspring[i])[0]

        # Parallel evaluation of new individuals
        invalid_indv = [indv for indv in offspring if not indv.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_indv)  # Parallel!
        for indv, fit in zip(invalid_indv, fitnesses):
            indv.fitness.values = fit

        # Replace population
        population[:] = elite + offspring

        # Update hall of fame and statistics
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=generation, nevals=len(invalid_indv), **record)

        # Print progress
        if generation % 10 == 0:
            print(f"Gen {generation:4d}: "
                  f"max={record['max']:.4f}, "
                  f"avg={record['avg']:.4f}, "
                  f"min={record['min']:.4f}, "
                  f"std={record['std']:.4f}")

        # Early stopping for perfect solution
        if record['max'] >= 0.99:  # 99% accuracy for MNIST
            print(f"\nExcellent solution found at generation {generation}!")
            generation += 1
            break

        generation += 1

    if interrupted:
        print(f"\nInterrupted at generation {generation}")
    else:
        print(f"\nEvolution completed after {generation} generations")

    best_ind = hof[0]
    print(f"Best fitness: {best_ind.fitness.values[0]:.4f}")

    return {
        'generations': generation,
        'best_fitness': best_ind.fitness.values[0],
        'best_individual': best_ind,
        'logbook': logbook,
        'final_population': population
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"MNIST Evolution with GEPNN")
    print(f"Population: {POPULATION_SIZE}, Cores: {CORES}")
    print(f"Head Length: {HEAD_LENGTH}, Weights: {NUM_WEIGHTS}, Biases: {NUM_BIASES}")
    print("=" * 60)

    try:
        # Run single experiment with parallel fitness evaluation
        result = run_single_ga()

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mnist_h{HEAD_LENGTH}_g{result['generations']}_f{result['best_fitness']:.4f}_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)

        # Convert to JSON-serializable format
        json_result = {
            'parameters': {
                'population_size': POPULATION_SIZE,
                'max_generations': MAX_GENERATION_LIMIT,
                'crossover_rate': CROSSOVER_RATE,
                'mutation_rate': MUTATION_RATE,
                'tournament_size': TOURNAMENT_SIZE,
                'elitism_ratio': ELITISM_RATIO,
                'head_length': HEAD_LENGTH,
                'num_weights': NUM_WEIGHTS,
                'num_biases': NUM_BIASES,
                'cores_used': CORES
            },
            'result': {
                'generations': result['generations'],
                'best_fitness': result['best_fitness'],
                'best_individual': {
                    'expression': result['best_individual'].export(),
                    'fitness': result['best_individual'].fitness.values[0]
                },
                'evolution_log': [dict(record) for record in result['logbook']]
            }
        }

        with open(filepath, 'w') as f:
            json.dump(json_result, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
    finally:
        # Clean up the pool
        pool.close()
        pool.join()
        print("Cleanup complete")
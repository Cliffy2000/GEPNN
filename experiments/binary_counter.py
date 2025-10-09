# individual - check
# operators (mutate) - check
# functions - check
# fitness - check

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
from evaluation.fitness import evaluate_binary_counter
from utils.output_utils import print_header, print_section_break

import signal

# ==================================================

# Experiment Parameters
ITERATIONS = 1
CORES = 1

# GA Parameters
POPULATION_SIZE = 250
MAX_GENERATION_LIMIT = 1000
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 2
ELITISM_RATIO = 0.02

# GEP Parameters
HEAD_LENGTH = 6
NUM_INPUTS = 1
NUM_WEIGHTS = 6    # all nodes apart from the root node need a weight
NUM_BIASES = 6     # all head nodes need a bias

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
    :param indv1: full individual object
    :param indv2: full individual object
    :return: Tuple of two new individuals
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
    :param indv: Full individual object
    :return: Tuple of shape `(new_indv, )`
    """
    mutated = mutate(indv, MUTATION_RATE)

    new_indv = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES,
        chromosome = mutated
    )

    new_indv.fitness = creator.FitnessMax()
    return (new_indv, )     # must return a tuple that contains the individual


creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("GEPIndividual", Individual, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("map", map)
toolbox.register("individual", create_individual_wrapper)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_binary_counter)
toolbox.register("select", tools.selRoulette)
toolbox.register("crossover", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)


def run_ga_iteration(iteration_num):
    """
    Fully executes a complete run of the genetic algorithm and yield its results
    :param iteration_num: Current iteration number for tracking
    :return: Dictionary with results from the iteration
    """
    global interrupted

    population = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(map(toolbox.evaluate, population))
    for indv, fit in zip(population, fitnesses):
        indv.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)
    hof.update(population)

    elite_size = int(ELITISM_RATIO * POPULATION_SIZE)
    generation = 0
    perfect_found = False

    print(f"Iteration {iteration_num}: Starting...")

    while generation < MAX_GENERATION_LIMIT:
        elite = tools.selBest(population, elite_size)

        offspring_size = POPULATION_SIZE - elite_size
        parents = toolbox.select(population, offspring_size)
        parents = list(map(toolbox.clone, parents))
        offspring = []

        for i in range(0, len(parents)-1, 2):
            if interrupted:
                break
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = toolbox.crossover(parents[i], parents[i+1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([toolbox.clone(parents[i]), toolbox.clone(parents[i+1])])

        if interrupted:
            break

        # odd number of parents
        if len(parents) % 2:
            offspring.append(toolbox.clone(parents[-1]))

        # apply mutation
        for i in range(len(offspring)):
            if np.random.random() < MUTATION_RATE:
                offspring[i] = toolbox.mutate(offspring[i])[0]

        # evaluates new individuals
        invalid_indv = [indv for indv in offspring if not indv.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_indv))
        for indv, fit in zip(invalid_indv, fitnesses):
            indv.fitness.values = fit

        population[:] = elite + offspring
        hof.update(population)
        record = stats.compile(population)

        if generation % 1 == 0 and generation > 0:
            print(f"  Gen {generation}: best={record['max']:.4f}, avg={record['avg']:.4f}")

        if record['max'] >= 0.9999:
            perfect_found = True
            generation += 1
            print(f"  Perfect solution found at generation {generation}!")
            break

        generation += 1

    best_ind = hof[0]

    if interrupted:
        print(f"  Interrupted at generation {generation}. Best fitness: {best_ind.fitness.values[0]:.4f}")
    elif not perfect_found:
        print(f"  Completed {generation} generations. Best fitness: {best_ind.fitness.values[0]:.4f}")

    return {
        'iteration': iteration_num,
        'generations': generation,
        'best_fitness': best_ind.fitness.values[0],
        'best_individual': best_ind,
        'perfect_found': perfect_found,
        'final_avg_fitness': record['avg']
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"T-XOR Evolution - {ITERATIONS} iterations")
    print(f"Population: {POPULATION_SIZE}, Cores: {CORES}")
    print("=" * 60)

    results = []

    try:
        results = [run_ga_iteration(ITERATIONS)]

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected!")
        results = []
    finally:
        print("=" * 60)
        perfect_count = sum(1 for r in results if r['perfect_found'])
        print(f"Completed: {perfect_count}/{ITERATIONS} perfect solutions")
        print(f"Saving results to JSON...")

        # Calculate accuracy (percentage of perfect solutions)
        perfect_count = sum(1 for r in results if r['perfect_found'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"binary_h{HEAD_LENGTH}_s{perfect_count}_n{ITERATIONS}_c{CROSSOVER_RATE}_m{MUTATION_RATE}_{timestamp}.json"

        filepath = os.path.join(os.path.dirname(__file__), filename)

        # Convert results to JSON-serializable format
        json_results = []
        for r in results:
            json_results.append({
                'iteration': r['iteration'],
                'generations': r['generations'],
                'best_fitness': r['best_fitness'],
                'perfect_found': r['perfect_found'],
                'final_avg_fitness': r['final_avg_fitness'],
                'best_individual': {
                    'expression': r['best_individual'].export(),
                    'fitness': r['best_individual'].fitness.values[0]
                }
            })

        with open(filepath, 'w') as f:
            json.dump({
                'parameters': {
                    'iterations': ITERATIONS,
                    'population_size': POPULATION_SIZE,
                    'max_generations': MAX_GENERATION_LIMIT,
                    'crossover_rate': CROSSOVER_RATE,
                    'mutation_rate': MUTATION_RATE,
                    'tournament_size': TOURNAMENT_SIZE,
                    'elitism_ratio': ELITISM_RATIO,
                    'head_length': HEAD_LENGTH
                },
                'results': json_results
            }, f, indent=2)

        print(f"Results saved to: {filepath}")

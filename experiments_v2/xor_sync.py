import sys
import os

# Reset CPU affinity (cluster/JupyterHub fix)
try:
    os.sched_setaffinity(0, set(range(os.cpu_count())))
except:
    pass

# Thread limiting for NumPy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
import argparse
import numpy as np
import signal
import multiprocessing as mp
from multiprocessing import Pool, Value
from deap import base, creator, tools
from tqdm import tqdm
from core.individual_v2 import Individual_v2_xor
from core.operators_v2 import crossover_sync, mutate_v2_xor
from evaluation.fitness_v2 import evaluate_xor

# ==================================================
# Default Parameters
# ==================================================

# Experiment Parameters
ITERATIONS = 100
CORES = 20

# GA Parameters
POPULATION_SIZE = 250
MAX_GENERATION_LIMIT = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.35
TOURNAMENT_SIZE = 2
ELITISM_RATIO = 0.04

# GEP Parameters
HEAD_LENGTH = 6
NUM_INPUTS = 2

# Output control
QUIET = False

# Shutdown event for graceful termination
shutdown_event = None
completed_count = None
success_count = None


# ==================================================


def parse_args():
    parser = argparse.ArgumentParser(description='XOR Evolution Experiment')

    parser.add_argument('--iterations', type=int)
    parser.add_argument('--cores', type=int)
    parser.add_argument('--pop_size', type=int)
    parser.add_argument('--max_gen', type=int)
    parser.add_argument('--crossover', type=float)
    parser.add_argument('--mutation', type=float)
    parser.add_argument('--elitism', type=float)
    parser.add_argument('--head', type=int)
    parser.add_argument('--quiet', action='store_true')

    return parser.parse_args()


def apply_args(args):
    global ITERATIONS, CORES, POPULATION_SIZE, MAX_GENERATION_LIMIT
    global CROSSOVER_RATE, MUTATION_RATE, ELITISM_RATIO, HEAD_LENGTH, QUIET

    if args.iterations is not None:
        ITERATIONS = args.iterations
    if args.cores is not None:
        CORES = args.cores
    if args.pop_size is not None:
        POPULATION_SIZE = args.pop_size
    if args.max_gen is not None:
        MAX_GENERATION_LIMIT = args.max_gen
    if args.crossover is not None:
        CROSSOVER_RATE = args.crossover
    if args.mutation is not None:
        MUTATION_RATE = args.mutation
    if args.elitism is not None:
        ELITISM_RATIO = args.elitism
    if args.head is not None:
        HEAD_LENGTH = args.head
    if args.quiet:
        QUIET = True


# Parse and apply arguments
apply_args(parse_args())

# Derived parameters
NUM_WEIGHTS = HEAD_LENGTH * NUM_INPUTS
NUM_BIASES = HEAD_LENGTH


def init_worker(event, counter, success):
    global shutdown_event, completed_count, success_count
    shutdown_event = event
    completed_count = counter
    success_count = success
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Reset CPU affinity for worker
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count())))
    except:
        pass


def create_individual_wrapper():
    indv = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES
    )
    indv.fitness = creator.FitnessMax()
    return indv


def crossover_wrapper(indv1, indv2):
    offspring1, offspring2 = crossover_sync(indv1, indv2, CROSSOVER_RATE)

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
    mutated = mutate_v2_xor(indv, MUTATION_RATE)

    new_indv = creator.GEPIndividual(
        head_length=HEAD_LENGTH,
        num_inputs=NUM_INPUTS,
        num_weights=NUM_WEIGHTS,
        num_biases=NUM_BIASES,
        chromosome=mutated
    )

    new_indv.fitness = creator.FitnessMax()
    return (new_indv,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("GEPIndividual", Individual_v2_xor, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("map", map)
toolbox.register("individual", create_individual_wrapper)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_xor)
toolbox.register("select", tools.selRoulette)
toolbox.register("crossover", crossover_wrapper)
toolbox.register("mutate", mutation_wrapper)


def run_ga_iteration(iteration_num):
    global shutdown_event

    population = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(toolbox.map(toolbox.evaluate, population))
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
    interrupted = False

    if not QUIET:
        print(f"Iteration {iteration_num}: Starting...")

    while generation < MAX_GENERATION_LIMIT:
        # Check for shutdown signal
        if shutdown_event is not None and shutdown_event.is_set():
            interrupted = True
            break

        elite = tools.selBest(population, elite_size)

        offspring_size = POPULATION_SIZE - elite_size
        parents = toolbox.select(population, offspring_size)
        parents = list(map(toolbox.clone, parents))
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            if shutdown_event is not None and shutdown_event.is_set():
                interrupted = True
                break
            child1, child2 = toolbox.crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])

        if interrupted:
            break

        if len(parents) % 2:
            offspring.append(toolbox.clone(parents[-1]))

        for i in range(len(offspring)):
            offspring[i] = toolbox.mutate(offspring[i])[0]

        invalid_indv = [indv for indv in offspring if not indv.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_indv)
        for indv, fit in zip(invalid_indv, fitnesses):
            indv.fitness.values = fit

        population[:] = elite + offspring
        hof.update(population)
        record = stats.compile(population)

        if not QUIET and generation % 100 == 0 and generation > 0:
            print(f"  Gen {generation}: best={record['max']:.4f}, avg={record['avg']:.4f}")

        if record['max'] >= 0.9999:
            perfect_found = True
            generation += 1
            if not QUIET:
                print(f"  Perfect solution found at generation {generation}!")
            break

        generation += 1

    best_ind = hof[0]

    if interrupted:
        print(f"  Iteration {iteration_num} interrupted at generation {generation}.")
    elif not perfect_found and not QUIET:
        print(f"  Completed {generation} generations. Best fitness: {best_ind.fitness.values[0]:.4f}")

    # Increment counters
    with completed_count.get_lock():
        completed_count.value += 1
    if perfect_found:
        with success_count.get_lock():
            success_count.value += 1

    return {
        'iteration': iteration_num,
        'generations': generation,
        'best_fitness': best_ind.fitness.values[0],
        'best_individual': best_ind,
        'perfect_found': perfect_found,
        'final_avg_fitness': record['avg'],
        'interrupted': interrupted
    }


if __name__ == "__main__":
    print("-" * 40)
    print(f"XOR Evolution - {ITERATIONS} iterations")
    print(f"Population: {POPULATION_SIZE}, Cores: {CORES}")
    print("-" * 40)

    results = []
    shutdown_event = mp.Event()
    completed_count = Value('i', 0)
    success_count = Value('i', 0)
    pool = None

    try:
        pool = Pool(processes=CORES, initializer=init_worker, initargs=(shutdown_event, completed_count, success_count))
        iteration_numbers = list(range(1, ITERATIONS + 1))
        async_result = pool.map_async(run_ga_iteration, iteration_numbers)

        # Progress bar
        pbar = tqdm(total=ITERATIONS, desc="Progress", unit="iter")
        last_completed = 0

        while not async_result.ready():
            async_result.wait(timeout=0.5)
            current = completed_count.value
            if current > last_completed:
                pbar.update(current - last_completed)
                last_completed = current
            pbar.set_postfix({"success": f"{success_count.value}/{completed_count.value}"})

        # Final update
        pbar.update(ITERATIONS - last_completed)
        pbar.set_postfix({"success": f"{success_count.value}/{ITERATIONS}"})
        pbar.close()

        results = async_result.get()

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected! Terminating workers...")

        if pool is not None:
            pool.terminate()
            pool.join()

        print("Workers terminated.")
        results = []
    finally:
        if pool is not None:
            pool.close()
            pool.join()

        print("-" * 40)

        if len(results) == ITERATIONS:
            perfect_count = sum(1 for r in results if r['perfect_found'])
            print(f"Completed: {perfect_count}/{ITERATIONS} perfect solutions")
            print(f"Saving results to JSON...")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xor_sync_0.5acc_h{HEAD_LENGTH}_s{perfect_count}_n{ITERATIONS}_c{CROSSOVER_RATE:.2f}_m{MUTATION_RATE:.2f}_{timestamp}.json"

            filepath = os.path.join(os.path.dirname(__file__), 'xor\\', filename)

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
        else:
            print(f"Interrupted: {len(results)}/{ITERATIONS} iterations completed. Results not saved.")
import torch
import random
import numpy as np
from deap import base, creator, tools
from core.individual import Individual_xor
from core.network import Network
from core.operators import crossover_one_point, mutate_xor
from experiments.trial_xor_prev import POPULATION_SIZE, NUM_INPUTS, NUM_WEIGHTS, NUM_BIASES
from utils.output_utils import print_header, print_section_break
from evaluation.fitness import evaluate_xor


# GA Parameters
POPULATION_SIZE = 100
GENERATION_COUNT = 1000
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 10

# GEP Parameters
HEAD_LENGTH = 10    # tail length = 10 * (2 - 1) + 1
NUM_INPUTS = 2
NUM_WEIGHTS = 20    # all nodes apart from the root node need a weight
NUM_BIASES = 10     # all head nodes need a bias

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

    new_indv = new_indv2 = creator.GEPIndividual(
        head_length = HEAD_LENGTH,
        num_inputs = NUM_INPUTS,
        num_weights = NUM_WEIGHTS,
        num_biases = NUM_BIASES,
        chromosome = mutated
    )

    new_indv.fitness = creator.FitnessMax()
    return (new_indv, )     # must return a tuple that contains the individual





import random
import numpy as np
from .individual import GepnnChromosome


class GepnnEvolution:
    def __init__(self, primitive_set, population_size=50, head_length=5, num_genes=3):
        self.primitive_set = primitive_set
        self.population_size = population_size
        self.head_length = head_length
        self.num_genes = num_genes

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = GepnnChromosome.create_random(
                self.primitive_set,
                self.head_length,
                self.num_genes
            )
            population.append(chromosome)
        return population

    def evaluate_population(self, population, fitness_evaluator):
        fitness_scores = []
        for chromosome in population:
            fitness = fitness_evaluator.evaluate(chromosome)
            fitness_scores.append(fitness)
        return fitness_scores

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_index])
        return selected

    def mutate_chromosome(self, chromosome, mutation_rate=0.1):
        for gene in chromosome.genes:
            for i in range(len(gene)):
                if random.random() < mutation_rate:
                    all_primitives = list(self.primitive_set.functions) + list(self.primitive_set.terminals)
                    gene[i] = random.choice(all_primitives)

    def crossover_chromosomes(self, parent1, parent2):
        child1_genes = []
        child2_genes = []

        for i in range(len(parent1.genes)):
            if random.random() < 0.5:
                child1_genes.append(parent1.genes[i][:])
                child2_genes.append(parent2.genes[i][:])
            else:
                child1_genes.append(parent2.genes[i][:])
                child2_genes.append(parent1.genes[i][:])

        child1 = GepnnChromosome(child1_genes, self.primitive_set)
        child2 = GepnnChromosome(child2_genes, self.primitive_set)
        return child1, child2

    def evolve(self, fitness_evaluator, num_generations=100, mutation_rate=0.1, crossover_rate=0.8):
        population = self.create_population()

        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(num_generations):
            fitness_scores = self.evaluate_population(population, fitness_evaluator)

            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.min(fitness_scores)
            best_index = np.argmin(fitness_scores)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            print(f"Generation {generation}: avg={avg_fitness:.4f}, best={best_fitness:.4f}")

            if generation == num_generations - 1:
                break

            selected = self.tournament_selection(population, fitness_scores)

            next_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

                if random.random() < crossover_rate:
                    child1, child2 = self.crossover_chromosomes(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                self.mutate_chromosome(child1, mutation_rate)
                self.mutate_chromosome(child2, mutation_rate)

                next_population.extend([child1, child2])

            population = next_population[:self.population_size]

        final_fitness = self.evaluate_population(population, fitness_evaluator)
        best_index = np.argmin(final_fitness)
        best_chromosome = population[best_index]

        return {
            'best_chromosome': best_chromosome,
            'best_fitness': final_fitness[best_index],
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }
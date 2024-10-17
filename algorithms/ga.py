import numpy as np
import random
from typing import List, Tuple, Callable
import csv
import time
from datetime import datetime
from utils.helpers import GALogger, GAMetrics, GAVisualizer

class Individual:
    def __init__(self, genes: dict):
        self.genes = genes
        self.fitness = None

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"

class GeneticAlgorithm:
    def __init__(self, gene_space: dict, fitness_func: Callable, population_size: int = 50, 
                 generations: int = 20, mutation_rate: float = 0.1, elite_size: int = 2, log_dir='ga_logs'):
        self.gene_space = gene_space
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.logger = GALogger(log_dir)
        self.visualizer = GAVisualizer(log_dir)
        self.metrics = GAMetrics()

    def initialize_population(self) -> List[Individual]:
        population = []
        for _ in range(self.population_size):
            genes = {key: random.choice(value) if isinstance(value, list) else random.uniform(value[0], value[1])
                     for key, value in self.gene_space.items()}
            population.append(Individual(genes))
        return population

    def selection(self, population: List[Individual]) -> List[Individual]:
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = population[:self.elite_size]
        non_elite = population[self.elite_size:]
        selected = elite + random.choices(non_elite, k=self.population_size - self.elite_size)
        return selected

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child_genes = {}
        for gene in self.gene_space.keys():
            if random.random() < 0.5:
                child_genes[gene] = parent1.genes[gene]
            else:
                child_genes[gene] = parent2.genes[gene]
        return Individual(child_genes)

    def mutation(self, individual: Individual) -> Individual:
        for gene, value in individual.genes.items():
            if random.random() < self.mutation_rate:
                if isinstance(self.gene_space[gene], list):
                    individual.genes[gene] = random.choice(self.gene_space[gene])
                else:
                    individual.genes[gene] = random.uniform(self.gene_space[gene][0], self.gene_space[gene][1])
        return individual

    def evolve(self) -> Tuple[Individual, List[float]]:
        population = self.initialize_population()
        best_fitness_history = []
        avg_fitness_history = []
        generations = []
        start_time = time.time()

        for generation in range(self.generations):
            fitnesses = []
            for individual in population:
                individual.fitness = self.fitness_func(individual.genes)
                fitnesses.append(individual.fitness)
                self.all_individuals.append((generation, individual))

            best_individual = max(population, key=lambda x: x.fitness)
            best_fitness = best_individual.fitness
            avg_fitness = np.mean(fitnesses)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(generation)

            diversity = self.metrics.diversity(population)
            improvement_rate = self.metrics.improvement_rate(best_fitness_history)

            self.logger.log_generation(generation, best_fitness, avg_fitness, best_individual)
            self.visualizer.plot_fitness_progress(generations, best_fitness_history, avg_fitness_history)
            self.visualizer.plot_parameter_distribution(self.all_individuals, generation)

            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")
            print(f"Diversity: {diversity}, Improvement Rate: {improvement_rate}")

            if self.metrics.convergence(best_fitness_history):
                print("Converged. Stopping early.")
                break

            if generation == self.generations - 1:
                break


            new_population = []
            new_population.extend(self.selection(population)[:self.elite_size])

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

        best_individual = max(population, key=lambda x: x.fitness)
        
        total_time = time.time() - start_time
        self.logger.log_final_results(best_individual, generation + 1, total_time)
        return best_individual, best_fitness_history
    
    def save_results(self, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Generation'] + list(self.gene_space.keys()) + ['Fitness']
            writer.writerow(headers)
            
            for generation, individual in self.all_individuals:
                row = [generation] + list(individual.genes.values()) + [individual.fitness]
                writer.writerow(row)
        
        print(f"Results saved to {filename}")

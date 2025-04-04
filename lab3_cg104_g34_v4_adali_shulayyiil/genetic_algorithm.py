import random
from typing import List, Tuple
from functions import init_ranges
import numpy as np
from numpy import ndarray
from numpy.random import choice


def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)
epsilon = 1e-100

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int, #N
        mutation_rate: float, #p_m
        mutation_strength: float, #s_m
        crossover_rate: float, #p_c
        num_generations: int, #G
        fitness_function: callable, #f
        

    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.fitness_function = fitness_function
        self.init_range = init_ranges[fitness_function]
    
    def initialize_population(self):
        self.population = []
      
        for i in range(self.population_size):
            individual = []
            for j in range(len(self.init_range)): 
                individual.append(random.uniform(self.init_range[j][0], self.init_range[j][1]))  
            
            self.population.append(individual)
        return self.population
    def evaluate_population(self, population) -> List[float]:
        
        fitness_values = []
        for individual in population:
            fitness_value = self.fitness_function(individual[0], individual[1])
            fitness_values.append(fitness_value)
        return fitness_values

    def selection(self, population:list, fitness_values:list) -> List[ndarray]:
        candidates = []
        candidate_fitness = []
        i = 0
        while (i<len(population)):
            if random.random() < self.crossover_rate:
                candidates.append(population[i])
                candidate_fitness.append(fitness_values[i])
                population.pop(i)
                fitness_values.pop(i)
            i += 1
        if len(candidates) == 0:
            return None        

        selected:List[ndarray] = []
        w_fitness = [1/(x+epsilon) for x in candidate_fitness] #account for x = 0
        
        w = sum(w_fitness)
        
        w = np.power(w, -1)
        
        w_fitness = [w/(x+epsilon) for x in candidate_fitness]
        for a in range(len(candidates)):
            
            draw = choice(len(candidates),2,True,w_fitness)
            selected.append([candidates[draw[0]],candidates[draw[1]]])
        

        return selected

    def crossover(self, parents) -> List[any]:
        if parents is None:
            return []
        offsprings = []
        for (p1,p2) in parents:
            
            alpha = random.uniform(0,1)
            offspring = (alpha*p1[0]+(1-alpha)*p2[0], alpha*p1[1]+(1-alpha)*p2[1])
            offsprings.append(offspring)
        
        return offsprings

    def mutate(self, individuals) -> List[any]:
        for individual in individuals:
            if random.random() < self.mutation_rate:
                delta = np.random.normal(0,self.mutation_strength, 2)
                individual = np.add(individual, delta)
            
        return individuals

    def evolve(self, seed: int):
        # Run the genetic algorithm and return the lists that contain the best solution for each generation,
        #   the best fitness for each generation and average fitness for each generation
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []
        fitness_values = self.evaluate_population(population)
        for generation in range(self.num_generations):
            
            
            parents_for_reproduction = self.selection(population, fitness_values)
            offspring = self.crossover(parents_for_reproduction)
            population = population + offspring
            population = self.mutate(population)
            fitness_values = self.evaluate_population(population)

            best_fitness = min(fitness_values)
            best_solution = population[fitness_values.index(best_fitness)]
            average_fitness = sum(fitness_values) / len(fitness_values)

            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            average_fitness_values.append(average_fitness)


           

        return best_solutions, best_fitness_values, average_fitness_values

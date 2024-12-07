import time
import matplotlib.pyplot as plt
from numpy.random import randint, rand
import pandas as pd
import numpy as np

class GeneticAlgorithmKnapsack:
    def __init__(
        self,
        file_name = None,
        maximize=True,
        population_size=100,
        tournament_size=4,
        crossover_prob=0.9,
        mutation_prob=0.02,
        elitism=False,
        max_time=60,
        max_gen=2000000000,
        january = None,
        initial_capital = 10000,
        operational_cost = 1
    ):
        self.file_name = file_name
        self.maximize = maximize
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.max_time = max_time
        self.max_gen = max_gen
        self.january = pd.read_csv(self.file_name) if self.file_name else self.generate_dataset
        self.initial_capital = initial_capital
        self.operational_cost = operational_cost
        self.population = list()

    @staticmethod
    def generate_dataset(
        num_stocks=500, start="2024-01-01", end="2024-01-31", freq="D"
    ):
        # Create a date range for all days in January
        date_range = pd.date_range(start=start, end=end, freq=freq)
        data = []
        for date in date_range:
            row = [date] + list(
                np.random.randint(100, 501, size=num_stocks)
            )
            data.append(row)
        columns = ["Date"] + [str(i) for i in range(1, num_stocks + 1)]
        df = pd.DataFrame(data, columns=columns)
        return df
    
    @staticmethod
    def aux_check_chromosome_has_previous_stocks(chromosome) :
        for stock in range(chromosome.shape[0]):
            for day in range(chromosome.shape[1]):
                value += chromosome[stock-1][day-1]
                if value < 0 : 
                    return False
        return True
            
        
    
    def check_chromosome(self,chromosome) :
        total = self.initial_capital
        day_sold = 0
        if np.any(chromosome[:, 0] < 0) : # Can´t sell in the first day 
            return False
        if not aux_check_chromosome_has_previous_stocks(chromosome) : # before selling stocks we must own them
            return False
        for active_index in range(chromosome.shape[1]): # Iterate through columns(days)
            day_sold,day_spent = 0,0
            if np.any(chromosome[:, active_index] > 1): # If it decides to buy
                day_spent = np.sum(chromosome[:, active_index] * self.january.iloc[active_index][active_index] )     # Pero que sume solo cuando es positivo chromosome[:, active_index]
                if day_spent > total  : # Cannot spend more than you have. We first buy, then we can sell.
                    return False
            if np.any(chromosome[:, active_index] < 1): # If it decides to sell
                day_sold = np.sum(chromosome[:, active_index] * self.january.iloc[active_index-1][active_index+1] ) # Pero que sume solo cuando es negativo chromosome[:, active_index]
            
            total += day_sold - day_spent
            if total < 0:    
                return False  
        return True
            
        
    @staticmethod
    def generate_chromosome(n_assets = 500, n_days = 31):
        valid = False
        while not valid :
            chromosome = np.random.randint(-10, 10, size=(n_days, n_assets))  # Cantidades aleatorias
            valid = check_chromosome(chromosome)
        return chromosome

    def knapsack(self, x, sizes, benefits, max_capacity):
        current_capacity = 0
        total_benefit = 0
        for i in range(len(x)):
            if x[i] == 1:
                if current_capacity + sizes[i] <= max_capacity:
                    current_capacity += sizes[i]
                    total_benefit += benefits[i]
        return total_benefit

    def is_better_than(self, a, b):
        return a > b if self.maximize else a < b

    def selection(self, population, fitness):
        chosen = randint(len(population))
        for i in randint(0, len(population), self.tournament_size - 1):
            if self.is_better_than(fitness[i], fitness[chosen]):
                chosen = i
        return population[chosen]

    def crossover(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.crossover_prob:
            point = randint(1, len(p1) - 2)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
        return [c1, c2]

    def mutation(self, chromosome):
        for i in range(len(chromosome)):
            if rand() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]

    def best_average_worst(self, fitness):
        best_chromosome = 0
        best_fitness = fitness[0]
        worst_fitness = fitness[0]
        average_fitness = sum(fitness) / self.population_size
        for i in range(len(fitness)):
            if self.is_better_than(fitness[i], best_fitness):
                best_chromosome = i
                best_fitness = fitness[i]
            if self.is_better_than(worst_fitness, fitness[i]):
                worst_fitness = fitness[i]
        return best_chromosome, best_fitness, average_fitness, worst_fitness

    def genetic_algorithm(self, n_bits, max_capacity, sizes, benefits):
        generation = 0
        evolution = []
        population = [
            randint(0, 2, n_bits).tolist() for _ in range(self.population_size)
        ]
        fitness = [self.knapsack(c, sizes, benefits, max_capacity) for c in population]
        best_chromosome, best_fitness, average_fitness, worst_fitness = (
            self.best_average_worst(fitness)
        )
        print(
            f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
        )
        evolution.append([best_fitness, average_fitness, worst_fitness])
        start_time = time.time()

        while generation < self.max_gen and time.time() - start_time < self.max_time:
            generation += 1
            selected = [
                self.selection(population, fitness) for _ in range(self.population_size)
            ]
            offspring = []
            if self.elitism:
                offspring.append(population[best_chromosome])
            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[(i + 1) % len(selected)]
                for c in self.crossover(p1, p2):
                    self.mutation(c)
                    if len(offspring) < self.population_size:
                        offspring.append(c)
            population = offspring
            fitness = [
                self.knapsack(c, sizes, benefits, max_capacity) for c in population
            ]
            best_chromosome, best_fitness, average_fitness, worst_fitness = (
                self.best_average_worst(fitness)
            )
            print(
                f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
            )
            evolution.append([best_fitness, average_fitness, worst_fitness])

        return population[best_chromosome], fitness[best_chromosome], evolution

    def execute(self):
        n_bits, max_capacity, sizes, benefits = self.read_knapsack_file()
        best_chromosome, best_fitness, evolution = self.genetic_algorithm(
            n_bits, max_capacity, sizes, benefits
        )
        print(
            f"Execution completed! Best solution: f({best_chromosome}) = {best_fitness:.3f}"
        )

        occupied_capacity = sum(
            sizes[i] for i in range(len(best_chromosome)) if best_chromosome[i] == 1
        )
        print(
            f"Maximum capacity: {max_capacity:.2f}, Occupied capacity: {occupied_capacity:.2f}"
        )
        plt.plot(range(len(evolution)), [x[0] for x in evolution], label="Best")
        plt.plot(range(len(evolution)), [x[1] for x in evolution], label="Average")
        plt.plot(range(len(evolution)), [x[2] for x in evolution], label="Worst")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    GA = GeneticAlgorithmKnapsack()
    df = GA.generate_dataset()
    df.to_csv("data/data.csv",index=None)
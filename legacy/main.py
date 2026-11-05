# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pygad
import numpy
import random

function_inputs = [4, -2, 3.5, 5, 6]
ran_old = [0, 1, 2, 3, 4]
ran_new = []
desired_output = 44


def fitness_func(solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness


def population(n):
    for i in range(0, n):
        ran_new = random.sample(ran_old, len(function_inputs))
    return ran_new


ga_instance = pygad.GA(num_generations=100,
                       sol_per_pop=population(1),
                       num_genes=5,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       mutation_type="random")

ga_instance.run()
ga_instance.plot_result()
print(ga_instance.best_solution())

from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
from itertools import groupby
import plotly.graph_objects as go

# Importing the data
Blade_data = np.genfromtxt('blade_data.csv', delimiter=',')
# print(Blade_data[0, 1])
# print(len(Blade_data[:, 0]))   # row represents the blade number column represents the blade weight
Total_blade = len(Blade_data[:, 0]) - 1  # Total number of blades calculation
# print(Total_blade)
n = Total_blade
r = 1  # radius of the disk in mm


# objective function
# Fitness function
# Mass moment (Residual unbalance) is calculated
def fitness(x):
    m_x = np.zeros(n)
    m_y = np.zeros(n)
    # print(x)
    # print(x[1])
    # Calculate the unbalance of the total system in iterative loop
    for j in range(0, n):
        m_x[j] = Blade_data[x[j], 1] * r * np.cos((2 * 180 * j) / n)
        m_y[j] = Blade_data[x[j], 1] * r * np.sin((2 * 180 * j) / n)
        M_x = np.sum(m_x)
        M_y = np.sum(m_y)
        res_un = np.sqrt(M_x ** 2 + M_y ** 2)
        # print (res_un)
        fit = 1 / res_un
    return fit


# Parent selection
# Roulette wheel selection is preferred for the parent selection
# 1. Fitness value is calculated for all the population
# 2. Relative fitness value is calculated
# 3. Probability of the population range is calculated from 0 to 1
# 4. Random number is selected from the range 0-1
# 5. Based on the random number the 2 parent is selected
def selection(pop, scores):
    # Fitness value calculation f(x_i)
    f = scores
    # print(pop)
    relative_fitness = []
    # Relative fitness calculation p(i)
    for k in range(n_pop):
        pi = f[k] / sum(f)
        relative_fitness.append(pi)  # P is the relative fitness function
    # print(p)
    # Probability weight calculation

    probability = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

    # print(probability)
    # Roulette based selection
    # Random selection between 0-1

    random = np.random.random(2)

    # Parent selection based on the random number

    parent = np.zeros(2)
    for i in range(len(random)):
        for j in range(n_pop):
            if random[i] < probability[j]:
                parent[i] = j
                break
    parent.astype(np.int64)
    return parent


# crossover two parents to create two offspring (Children)
# Recombination crossover operator

# Gets edges for parent1, parent2
def find_edges(parent1, parent2):
    parent1_edges = calc_edges(parent1)
    parent2_edges = calc_edges(parent2)
    merged_edges = merge_edges(parent1_edges, parent2_edges)

    return parent1_edges, parent2_edges, merged_edges


# calculates edges for an individual
def calc_edges(individual):
    edges = []

    for position in range(len(individual)):
        if position == 0:
            edges.append([individual[position], (individual[-1], individual[position + 1])])
        elif position < len(individual) - 1:
            edges.append([individual[position], (individual[position - 1], individual[position + 1])])
        else:
            edges.append([individual[position], (individual[position - 1], individual[0])])

    return edges


# sort the edges
def sort_edges(individual):
    # individual.sort(lambda x, y: cmp(x[0],y[0]))
    individual.sort()
    # print(individual)


# perform an union on two parents
def merge_edges(parent1, parent2):
    sort_edges(parent1)
    sort_edges(parent2)

    edges = []
    for val in range(len(parent1)):
        edges.append([parent1[val][0], union(parent1[val][1], parent2[val][1])])

    return edges


# part of merge_edges - unions 2 individual
def union(individual1, individual2):
    edges = list(individual1)

    for val in individual2:
        if val not in edges:
            edges.append(val)
    return edges


# Edge recombination operator
def crossover(parent1, parent2, edges):
    k = []
    previous = None
    current = random.choice([parent1[0], parent2[0]])

    while True:
        k.append(current)

        if len(k) == len(parent1):
            break

        previous = remove_node_from_neighbouring_list(current, edges)
        current_neighbour = get_current_neighbour(previous, edges)

        next_node = None
        if len(current_neighbour) > 0:
            next_node = get_best_neighbour(current_neighbour)
        else:
            next_node = get_next_random_neighbour(k, edges)

        current = next_node[0]
    return k


# returns the best possible neighbour
def get_best_neighbour(neighbour):
    if len(neighbour) == 1:
        return neighbour[0]
    else:
        group_neighbour = group_neighbours(neighbour)
        return random.choice(group_neighbour[0])[1]


# part of get_best_neighbour
def group_neighbours(neighbours):
    sorted_neighbours = []

    # store length of each individual neighbour + neighbour in a list
    for neighbour in neighbours:
        sorted_neighbours.append((len(neighbour[1]), neighbour))

    # sort the new list
    sort_edges(sorted_neighbours)

    # group the neighbour by their size
    groups = []
    for k, g in groupby(sorted_neighbours, lambda x: x[0]):
        groups.append(list(g))

    return groups


# returns a random neighbour from remaining_edges that does not exist in current_path
def get_next_random_neighbour(current_path, remaining_edges):
    random_node = None

    while random_node is None:
        tmp_node = random.choice(remaining_edges)

        if tmp_node[0] not in current_path:
            random_node = tmp_node

    return random_node


# removes node from neighbouring list
def remove_node_from_neighbouring_list(node, neighbour_list):
    removed_node = None

    for n in neighbour_list:
        if n[0] == node:
            removed_node = n
            neighbour_list.remove(n)

        if node in n[1]:
            n[1].remove(node)

    return removed_node


# return neighbours for a give node(s)
def get_current_neighbour(nodes, neighbour_lists):
    neighbours = []

    if nodes is not None:
        for node in nodes[1]:
            for neighbour in neighbour_lists:
                if node == neighbour[0]:
                    neighbours.append(neighbour)

    return neighbours


# mutation operator
def mutation(p1, p2, pm, pM):
    # Mutation probability calculator
    if pM <= pm:
        exchange_position1 = random.sample(range(0, len(p1)), 2)
        p1_copy = p1.copy()
        p1[exchange_position1[0]] = p1_copy[exchange_position1[1]]
        p1[exchange_position1[1]] = p1_copy[exchange_position1[0]]
        exchange_position2 = random.sample(range(0, len(p1)), 2)
        p2_copy = p2.copy()
        p2[exchange_position2[0]] = p2_copy[exchange_position2[1]]
        p2[exchange_position2[1]] = p2_copy[exchange_position2[0]]


# genetic algorithm
def genetic_algorithm(fitness, n_iter, n_pop, pc, pm):
    # initial population of random bitstring
    blade = []
    change = 5000
    for i in range(1, n + 1):
        blade.append(i)
    # print(blade)
    pop = [random.sample(blade, len(blade)) for _ in range(n_pop)]
    # print(pop)
    # keep track of best solution
    best, best_eval = 100, fitness(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [fitness(c) for c in pop]

        # check for new best solution
        for i in range(len(pop)):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.8f" % (gen, pop[i], 1 / scores[i]))
        # select parents
        parent = selection(pop, scores)

        parent.astype(np.int64)

        # create the next generation
        # children = list()
        x = parent[0]
        y = parent[1]
        parent1 = pop[int(x)]
        parent2 = pop[int(y)]

        # crossover and mutation
        pC = np.random.random(1)
        pM = np.random.random(1)
        new_population = []
        if pC <= pc:
            matrix = find_edges(parent1, parent2)

            # Create child from the 2 Parent
            child = crossover(parent1, parent2, matrix[2])
            # pop.append(child)

            # Sorting the population
            # pop.sort(key=fitness, reverse=True)
            pop[len(pop) - 1] = child
            # pop.sort(key=fitness, reverse=True)
            # print(parent1)

            # extra code addition temporary

            # parent1 = child

            # perform mutate
            mutation(parent1, parent2, pm, pM)
            pp = np.random.random(1)
            popposition = random.sample(range(0, len(pop) - 1), 1)
            strings = [str(popposition) for popposition in popposition]
            a_string = "".join(strings)
            an_integer = int(a_string)
            # print(popposition)
            # print(pop[2])
            if pp < pm:
                pop[an_integer] = parent1
                # pop.append(parent1)
            else:
                pop[an_integer] = parent2

            # pop.sort(key = fitness, reverse=True)
            # pop[n_pop-3] = parent2
            # print(parent1)

            # if gen > change < 40000:
            #    new_popn = n_pop / 2
            #    for new_popp in range(int(new_popn)):
            #        new_pop = random.sample(blade, len(blade))
            #        pop.append(new_pop)

            #    change = 5000 + change
            #    print(pop)

    return [best, best_eval]


# define the total iterations
n_iter = 100
# define the population size
n_pop = 25
# crossover probability
pc = 0.95
# mutation probability
pm = 0.7
# perform the genetic algorithm search
best, score = genetic_algorithm(fitness, n_iter, n_pop, pc, pm)
print('Done!')
print('f(%s) = %f' % (best, score))

# Plotting of graph

radius = []
angle = []
breath = []
color = []

for i in range(len(best)):
    radius.append(Blade_data[best[i], 1])
    angle.append(i * 2 * 180 / len(best))
    breath.append(8)
    if (i % 2) == 0:
        colour = '#85EBFF'
    else:
        colour = '#405CFF'

    color.append(colour)

fig = go.Figure(go.Barpolar(
    r=radius,
    theta=angle,
    width=breath,
    marker_color=color,
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))

fig.update_layout(
    template=None,
    polar=dict(
        radialaxis=dict(range=[0, 2000], showticklabels=False, ticks=''),
        angularaxis=dict(showticklabels=False, ticks='')
    )
)

fig.show()

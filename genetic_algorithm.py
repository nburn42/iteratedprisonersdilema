import json
import multiprocessing
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from agents import TitForTat, Mac, Cynic, Random, Rube, Troll, Binomial, AdvancedPredict, PatternMatcher, IForgiveYou, \
    EricTheEvil, TitForTwoTats, GrimTrigger, Stephanie
from utils import AgentState, Decision, Agent, play_iterated_prisoners_dilemma


class Nathan(Agent):
    def __init__(self, genes, number):
        # numpy array of 50 genes
        self.genes = genes
        self.number = number
        self.fitness = -1

    def get_name(self):
        return 'Nathan ' + str(self.number)

    def set_fitness(self, fitness):
        self.fitness = fitness

    @property
    def initial_state(self) -> AgentState:
        # take first 10 genes
        return self.genes

    def weighted_choice(self, p):
        return [Decision.COOPERATE, Decision.DEFECT][int(random.random() < p)]

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        defectChance = self.genes[0]
        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return self.weighted_choice(defectChance), self.genes

        history = [d == Decision.DEFECT for d in other_agents_decisions]

        total_defections = sum(history)
        average = np.average(history)
        weighted_average = np.average(history, weights=np.arange(1, num_rounds + 1) * self.genes[1])
        recent_defect_window = int(100 * self.genes[2])
        recent_average = np.average(history[:recent_defect_window])
        last_defect = history[0]

        first_layer = np.array([num_rounds, total_defections, average, weighted_average, recent_average, last_defect])

        def neuron(features, params_offset):
            value = sum(features * self.genes[params_offset:params_offset + 6])
            if value > self.genes[params_offset + 6]:
                return value
            return 0

        second_layer = np.array([
            neuron(first_layer, 3),
            neuron(first_layer, 9),
            neuron(first_layer, 15),
            neuron(first_layer, 21),
            neuron(first_layer, 27),
            neuron(first_layer, 33),
        ])

        defectChance += neuron(first_layer, 3) * self.genes[40]
        defectChance += neuron(first_layer, 8) * self.genes[41]
        defectChance += neuron(first_layer, 13) * self.genes[42]
        defectChance += neuron(first_layer, 18) * self.genes[43]
        defectChance += neuron(first_layer, 23) * self.genes[44]
        defectChance += neuron(first_layer, 28) * self.genes[45]

        defectChance += neuron(second_layer, 33) * self.genes[46]

        return self.weighted_choice(defectChance), self.genes


def evaluate_agent(name, agent, opponents, expected_number_of_interactions):
    results = []
    for opponent in opponents:
        match = play_iterated_prisoners_dilemma(
            agent_1=agent,
            agent_2=opponent,
            expeted_number_of_interactions=expected_number_of_interactions,
        )
        results.append(match)
    return (name, sum([result[0] for result in results]))


def evaluate_population(population, expected_number_of_interactions, best_agent):
    """ Evaluate population """
    agents = []
    agent_names = []

    for nathan in population:
        agents.append(nathan)
        agent_names.append(nathan.get_name())

    opponents = [TitForTat(), Mac(), Cynic(), Random(random_seed=1), Rube(), Troll(), Binomial(), AdvancedPredict(),
               PatternMatcher(), IForgiveYou(), AdvancedPredict(), EricTheEvil(), TitForTwoTats(), GrimTrigger(),
               Stephanie(), best_agent]
    opponent_names = ['TitForTat', 'Mac', 'Cynic', 'Random', 'Rube', 'Troll', 'Binomial', 'Advanced', 'Matcher',
                   'Forgiver', 'AdvancedPredict', 'EricTheEvil', 'TitForTwoTats', 'GrimTrigger', 'Stephanie',
                   best_agent.get_name() + ' best']

    with multiprocessing.Pool() as pool:
        results = pool.starmap(evaluate_agent,
                               [(name, agent, opponents, expected_number_of_interactions) for agent, name in
                                zip(agents + opponents, agent_names + opponent_names)])

    agg_results = []

    # results = []

    # for first_agent, agent_name in zip(agents, agent_names):
    #     results.append([])
    #     agg_results.append([agent_name, 0])
    #     # if "Nathan" not in agent_name:
    #     #     continue
    #     for opponent in agents:
    #         match = play_iterated_prisoners_dilemma(
    #             agent_1=first_agent,
    #             agent_2=opponent,
    #             expeted_number_of_interactions=expected_number_of_interactions,
    #         )
    #         results[-1].append(match)
    #         agg_results[-1][1] = agg_results[-1][1] + results[-1][-1][0]
    #     agg_results[-1][1] = round(agg_results[-1][1], 2)

    agg_results = []
    for i, agent_name in enumerate(agent_names + opponent_names):
        agg_result = [agent_name, 0]
        for result in results:
            if result[0] == agent_name:
                agg_result[1] += result[1]
        agg_result[1] = round(agg_result[1], 2)
        agg_results.append(agg_result)

    # Set fitness
    for i in range(len(population)):
        population[i].set_fitness(agg_results[i][1])
    print(*sorted(agg_results, key=lambda x: x[1], reverse=True), sep='\n')


def select(population):
    # Round Robin Tournament
    winners = []

    # keep best
    winners.append(max(population, key=lambda x: x.fitness))

    index = 0
    value = 0
    worst = min(population, key=lambda x: x.fitness).fitness
    while len(winners) < 10:
        value += random.random() * (population[0].fitness - worst) * 2
        while value > (population[index].fitness - worst):
            value -= population[index].fitness - worst
            index += 1
            if index == len(population):
                index = 0
        winners.append(population[index])

    return winners


def breed(population, ii, mutation_rate=0.25):
    """ Breed next generation """
    # Initialize next generation
    next_generation = []

    # keep best
    next_generation.append(max(population, key=lambda x: x.fitness))

    # Repeat until next generation is complete
    while len(next_generation) < 20:
        # Select parents
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        parents = [parent1, parent2]

        choice_values = np.random.randint(0, 2, len(parents[0].genes))
        child_genes = np.zeros(len(parents[0].genes))
        for i in range(len(child_genes)):
            child_genes[i] = parents[choice_values[i]].genes[i] + np.random.normal(0, 1) * mutation_rate
        next_generation.append(Nathan(child_genes, ii + len(next_generation)))
    return next_generation


def genetic_algorithm():
    """ Genetic algorithm """
    # Initialize population
    population = [Nathan((np.random.rand(50) - 0.5), i) for i in range(10)]

    best = population[0]
    # Evaluate population
    evaluate_population(population, 50, best)
    # Repeat until termination condition met

    best_chart = []

    ii = 50
    while True:
        # Select parents
        population = select(population)
        # Breed next generation
        population = breed(population, ii)
        # Evaluate population
        evaluate_population(population, (500.0 * random.random()) + 50, best)
        ii += 50
        best = max(population, key=lambda x: x.fitness)

        print('Generation: {}'.format(ii))
        print('Best fitness: {}'.format(best.fitness))
        print('Best genes: {}'.format(best.genes))
        print('Best name: {}'.format(best.get_name()))

        # Save best solution
        with open('best.json', 'w') as f:
            json.dump(best.genes.tolist(), f)

        best_chart.append(best.fitness)

        # Save best chart matplotlib
        plt.plot(best_chart)
        plt.savefig('best_chart.png')
        plt.clf()


if __name__ == '__main__':
    genetic_algorithm()

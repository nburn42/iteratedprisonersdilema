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
        defect_chance = self.genes[0]
        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return self.weighted_choice(defect_chance), self.genes

        history = [d == Decision.DEFECT for d in other_agents_decisions]

        total_defections = sum(history)
        average = np.average(history)
        weighted_average_1 = np.average(history, weights=np.arange(1, num_rounds + 1) + self.genes[1])
        weighted_average_2 = np.average(history, weights=np.arange(1, num_rounds + 1) + self.genes[2])
        window_1 = max(abs(int(100 * self.genes[3])), 2)
        window_2 = max(abs(int(100 * self.genes[4])), 2)
        recent_average_1 = np.average(history[:window_1])
        recent_average_2 = np.average(history[:window_2])
        last_defect_1 = history[0]
        last_defect_2 = 0
        last_defect_3 = 0
        last_defect_4 = 0
        if len(history) > 3:
            last_defect_4 = history[3]
        elif len(history) > 2:
            last_defect_3 = history[2]
        elif len(history) > 1:
            last_defect_2 = history[1]

        first_layer = np.array([num_rounds, total_defections, average, weighted_average_1, weighted_average_2,
                                recent_average_1, recent_average_2, last_defect_1, last_defect_2, last_defect_3,
                                last_defect_4])

        p_count = 5
        def neuron(features, params_offset):
            value = sum(features * self.genes[params_offset:params_offset + len(features)]) + \
                    self.genes[params_offset + len(features)]

            if value > 0:
                return value
            return 0

        second_layer = np.array([
            neuron(first_layer, (i * len(first_layer) + p_count))
            for i in range(len(first_layer) * 3)
        ])
        p_count += (len(first_layer) + 1) * len(first_layer) * 3

        third_layer = neuron(second_layer, p_count)

        p_count += len(second_layer) + 1

        defect_chance += sum(first_layer * self.genes[p_count:p_count + len(first_layer)])
        p_count += len(first_layer)

        defect_chance += sum(second_layer * self.genes[p_count:p_count + len(second_layer)])
        p_count += len(second_layer)

        defect_chance += third_layer * self.genes[p_count]

        return self.weighted_choice(defect_chance), self.genes


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
    output_dir = '/home/nathan/ipd_output/overwork1/'

    # Initialize population
    population = [Nathan((np.random.rand(480) - 0.5), i) for i in range(500)]

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
        ii += len(population)
        best = max(population, key=lambda x: x.fitness)

        print('Generation: {}'.format(ii))
        print('Best fitness: {}'.format(best.fitness))
        print('Best genes: {}'.format(best.genes))
        print('Best name: {}'.format(best.get_name()))

        # Save best solution
        with open(output_dir + 'best.json', 'w') as f:
            json.dump(best.genes.tolist(), f)

        best_chart.append(best.fitness)

        # Save best chart matplotlib
        plt.plot(best_chart)
        plt.savefig(output_dir + 'best_chart.png')
        plt.clf()


if __name__ == '__main__':
    genetic_algorithm()

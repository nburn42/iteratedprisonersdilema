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
        defectChance = previous_state[0]
        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return self.weighted_choice(defectChance), previous_state

        history = [d == Decision.DEFECT for d in other_agents_decisions]

        average = np.average(history)
        weighted_average = np.average(history, weights=np.arange(1, num_rounds + 1) * previous_state[1])
        recent_defects = np.sum(history[:int(100 * previous_state[2])])
        last_defect = history[0]

        if average > previous_state[3]:
            defectChance += previous_state[4] + (average * previous_state[5]) + (weighted_average * previous_state[6]) + (
                    recent_defects * previous_state[7]) + (last_defect * previous_state[8])
        if average > previous_state[9]:
            defectChance += previous_state[10] + (average * previous_state[11]) + (weighted_average * previous_state[12]) + (
                    recent_defects * previous_state[13]) + (last_defect * previous_state[14])
        if average > previous_state[15]:
            defectChance += previous_state[16] + (average * previous_state[17]) + (weighted_average * previous_state[18]) + (
                    recent_defects * previous_state[19]) + (last_defect * previous_state[20])
        if weighted_average > previous_state[21]:
            defectChance += previous_state[22] + (average * previous_state[23]) + (weighted_average * previous_state[24]) + (
                    recent_defects * previous_state[25]) + (last_defect * previous_state[26])
        if weighted_average > previous_state[27]:
            defectChance += previous_state[28] + (average * previous_state[29]) + (weighted_average * previous_state[30]) + (
                    recent_defects * previous_state[31]) + (last_defect * previous_state[32])
        if recent_defects / 100 > previous_state[33]:
            defectChance += previous_state[34] + (average * previous_state[35]) + (weighted_average * previous_state[36]) + (
                    recent_defects * previous_state[37]) + (last_defect * previous_state[38])
        if last_defect:
            defectChance += previous_state[39] + (average * previous_state[40]) + (weighted_average * previous_state[41]) + (
                    recent_defects * previous_state[42]) + (last_defect * previous_state[43])

        return self.weighted_choice(defectChance), previous_state


def evaluate_population(population):
    """ Evaluate population """
    agents = []
    agent_names = []

    for nathan in population:
        agents.append(nathan)
        agent_names.append(nathan.get_name())

    agents += [TitForTat(), Mac(), Cynic(), Random(random_seed=1), Rube(), Troll(), Binomial(), AdvancedPredict(),
              PatternMatcher(), IForgiveYou(), AdvancedPredict(), EricTheEvil(), TitForTwoTats(), GrimTrigger(),
              Stephanie()]
    agent_names += ['TitForTat', 'Mac', 'Cynic', 'Random', 'Rube', 'Troll', 'Binomial', 'Advanced', 'Matcher',
                   'Forgiver', 'AdvancedPredict', 'EricTheEvil', 'TitForTwoTats', 'GrimTrigger', 'Stephanie']

    results = []
    agg_results = []

    expected_number_of_interactions = (2000.0 * random.random()) + 50
    for first_agent, agent_name in zip(agents, agent_names):
        results.append([])
        agg_results.append([agent_name, 0])
        # if "Nathan" not in agent_name:
        #     continue
        for opponent in agents:
            match = play_iterated_prisoners_dilemma(
                agent_1=first_agent,
                agent_2=opponent,
                expeted_number_of_interactions=expected_number_of_interactions,
            )
            results[-1].append(match)
            agg_results[-1][1] = agg_results[-1][1] + results[-1][-1][0]
        agg_results[-1][1] = round(agg_results[-1][1], 2)

    # Set fitness
    for i in range(len(population)):
        population[i].set_fitness(agg_results[i][1])
    # print(results)
    print(*sorted(agg_results, key=lambda x: x[1], reverse=True), sep='\n')
    # df = pd.DataFrame(results, columns=agent_names, index=agent_names)
    # print(df)


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


def breed(population, ii, mutation_rate=0.01):
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

        choice_values = np.random.randint(0, 2, 50)
        child_genes = np.zeros(50)
        for i in range(20):
            child_genes[i] = parents[choice_values[i]].genes[i] + np.random.normal(0, 1) * mutation_rate
        next_generation.append(Nathan(child_genes, ii + len(next_generation)))
    return next_generation


def genetic_algorithm():
    """ Genetic algorithm """
    # Initialize population
    population = [Nathan(0.1 * (np.random.rand(50) - 0.5), i) for i in range(100)]
    # Evaluate population
    evaluate_population(population)
    # Repeat until termination condition met

    best_chart = []

    ii = 50
    while True:
        # Select parents
        population = select(population)
        # Breed next generation
        population = breed(population, ii)
        # Evaluate population
        evaluate_population(population)
        ii += 50
        best = max(population, key=lambda x: x.fitness)

        print('Generation: {}'.format(ii))
        print('Best fitness: {}'.format(best.fitness))
        print('Best genes: {}'.format(best.genes))
        print('Best name: {}'.format(best.get_name()))

        # Save best solution
        with open('best.npy', 'wb') as f:
            np.save(f, best.genes)

        best_chart.append(best.fitness)

        # Save best chart matplotlib
        plt.plot(best_chart)
        plt.savefig('best_chart.png')
        plt.clf()





if __name__ == '__main__':
    genetic_algorithm()

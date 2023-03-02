import json
import multiprocessing
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from NathanExploit import NathanSelfPlay
from NathanGenetic import NathanGenetic
from agents import TitForTat, Mac, Cynic, Random, Rube, Troll, Binomial, AdvancedPredict, PatternMatcher, IForgiveYou, \
    TitForTwoTats, GrimTrigger, Stephanie, Konstantin, TribalPolitician, EricTheEvil1, EricTheEvil2, TribalCultist, \
    TribalCheater, EricTheEvil3, Vishal
from utils import play_iterated_prisoners_dilemma


def evaluate_agent(agent, opponents, expected_number_of_interactions):
    results = []
    for opponent in opponents:
        match = play_iterated_prisoners_dilemma(
            agent_1=agent,
            agent_2=opponent,
            expeted_number_of_interactions=expected_number_of_interactions,
        )
        results.append(match)
    return (agent.get_name(), sum([result[0] for result in results]))


def evaluate_population(agents, expected_number_of_interactions, best_agent, all_time_best_agent, self_play):
    """ Evaluate population """
    opponents = [TitForTat(), Mac(), Cynic(), Random(random_seed=1), Rube(), Troll(), Binomial(),
                 PatternMatcher(), IForgiveYou(), AdvancedPredict(), EricTheEvil1(), EricTheEvil2(), EricTheEvil3(),
                 TitForTwoTats(), GrimTrigger(), Stephanie(), TribalPolitician(), TribalCultist(),
                 TribalCheater(), Konstantin(), Vishal(), NathanSelfPlay(), best_agent, all_time_best_agent]
    players = agents + opponents
    if self_play:
        opponents += agents


    with multiprocessing.Pool() as pool:
        results = pool.starmap(evaluate_agent,
                               [(agent, opponents, expected_number_of_interactions) for agent in players])

    agg_results = []
    for i, agent_name in enumerate([a.get_name() for a in players]):
        agg_result = [agent_name, 0]
        for result in results:
            if result[0] == agent_name:
                agg_result[1] += result[1]
        agg_result[1] = round(agg_result[1], 2)
        agg_results.append(agg_result)

    # Set fitness
    for i in range(len(agents)):
        agents[i].set_fitness(agg_results[i][1])
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


def breed(population, ii, mutation_rate=0.05):
    """ Breed next generation """
    # Initialize next generation
    next_generation = []

    # keep best
    next_generation.append(max(population, key=lambda x: x.fitness))

    # Repeat until next generation is complete
    while len(next_generation) < 25:
        # Select parents
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        parents = [parent1, parent2]

        choice_values = np.random.randint(0, 2, len(parents[0].genes))
        child_genes = np.zeros(len(parents[0].genes))
        for i in range(len(child_genes)):
            child_genes[i] = parents[choice_values[i]].genes[i] + np.random.normal(0, 1) * mutation_rate
        next_generation.append(NathanGenetic(child_genes, ii + len(next_generation)))
    return next_generation


def genetic_algorithm(self_play=False):
    """ Genetic algorithm """
    output_dir = '/home/nathan/ipd_output/got_to_beat_vishal/'
    # output_dir = '/home/nburn42/ipd_output/self_play/'

    # Initialize population
    population = [NathanGenetic(None, i) for i in range(10)]

    best = TitForTat()
    all_time_best = TitForTat()
    all_time_best_fitness = 0
    evaluate_population(population, 50, best, all_time_best, self_play)

    best_chart = []

    ii = 50
    while True:
        ii += len(population)
        population = select(population)
        # Breed next generation
        population = breed(population, ii)
        # Evaluate population
        evaluate_population(population, (500.0 * random.random()) + 50, best, all_time_best, self_play)

        best = max(population, key=lambda x: x.fitness)
        best = NathanGenetic(best.genes, best.number + 'B', fitness=best.fitness)

        print('Generation: {}'.format(ii))
        print('Best fitness: {}'.format(best.fitness))
        # print('Best genes: {}'.format(best.genes))
        print('Best name: {}'.format(best.get_name()))

        # Save best solution
        with open(output_dir + 'best.json', 'w') as f:
            json.dump(best.genes.tolist(), f)

        if best.fitness > all_time_best_fitness:
            all_time_best = NathanGenetic(best.genes, best.number + 'AT', fitness=best.fitness)
            all_time_best_fitness = best.fitness
            with open(output_dir + f'all_time_best_{all_time_best_fitness}.json', 'w') as f:
                json.dump({
                    'fitness': all_time_best_fitness,
                    'name': all_time_best.get_name(),
                    'genes': all_time_best.genes.tolist(),
                    'time': str(datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
                }, f)

        best_chart.append((best.fitness, all_time_best_fitness))

        # Save best chart matplotlib
        plt.plot(best_chart)
        plt.savefig(output_dir + 'best_chart.png')
        plt.clf()


if __name__ == '__main__':
    genetic_algorithm(self_play=False)

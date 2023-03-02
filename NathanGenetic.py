import json
import os
import random
from typing import Tuple

import numpy as np

from utils import Agent, AgentState, Decision


class NathanGenetic(Agent):
    def __init__(self, genes, number, fitness=-1):
        # numpy array of 50 genes
        self.genes = genes
        if self.genes is None:
            # check if genes are in file
            filename = '/home/nathan/ipd_output/night1/all_time_best_59.79.json'
            # filename = '/home/nathan/iteratedprisonersdilema/ipd_output/overwork3/all_time_best_50.32.json'
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.genes = np.array(data['genes'])
            else:
                self.genes = (np.random.rand(838) - 0.5)

        self.number = str(number)
        self.fitness = fitness

    def get_name(self):
        return 'Nathan ' + str(self.number)

    def set_fitness(self, fitness):
        self.fitness = fitness

    @property
    def initial_state(self) -> AgentState:
        return None

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

        location_0 = 0
        location_1 = 1
        location_2 = int(100 * self.genes[5])
        location_3 = int(100 * self.genes[6])
        location_4 = int(100 * self.genes[7])
        location_5 = int(100 * self.genes[8])
        location_6 = int(100 * self.genes[9])
        location_7 = int(100 * self.genes[10])
        last_defect_0 = history[location_0]
        last_defect_1 = 0
        if len(history) > abs(location_1):
            last_defect_1 = history[abs(location_1)]
        last_defect_2 = 0
        if len(history) > abs(location_2):
            last_defect_2 = history[abs(location_2)]
        last_defect_3 = 0
        if len(history) > abs(location_3):
            last_defect_3 = history[abs(location_3)]
        last_defect_4 = 0
        if len(history) > abs(location_4):
            last_defect_4 = history[abs(location_4)]
        last_defect_5 = 0
        if len(history) > abs(location_5):
            last_defect_5 = history[abs(location_5)]
        last_defect_6 = 0
        if len(history) > abs(location_6):
            last_defect_6 = history[abs(location_6)]
        last_defect_7 = 0
        if len(history) > abs(location_7):
            last_defect_7 = history[abs(location_7)]

        first_layer = np.array([min(num_rounds, 200), total_defections, average, weighted_average_1, weighted_average_2,
                                recent_average_1, recent_average_2, last_defect_0, last_defect_1, last_defect_2,
                                last_defect_3,
                                last_defect_4, last_defect_5, last_defect_6, last_defect_7])

        p_count = 11

        def neuron(features, params_offset):
            try:
                value = sum(features * self.genes[params_offset:params_offset + len(features)]) + \
                        self.genes[params_offset + len(features)]
                return max(value, 0)
            except:
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
        # print(defect_chance, p_count)

        return self.weighted_choice(defect_chance), self.genes


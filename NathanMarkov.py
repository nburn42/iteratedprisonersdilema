from copy import deepcopy
from typing import Tuple

import numpy as np

from agents import TitForTat, Mac, Cynic, Random, Rube, Troll, Binomial, AdvancedPredict, PatternMatcher, IForgiveYou, \
    TitForTwoTats, GrimTrigger, Stephanie, TribalCheater, Konstantin, TribalPolitician, EricTheEvil1, EricTheEvil2, \
    EricTheEvil3, TribalCultist
from utils import Agent, AgentState, Decision, payoff


class NathanModelState:
    def __init__(self, agent_probabilities, my_history=None, their_history=None):
        self.my_history = my_history if my_history is not None else []
        self.their_history = their_history if their_history is not None else []
        self.agent_probabilities = agent_probabilities
        for agent in self.agent_probabilities:
            agent.predict_defect_percent(self)

    def clone(self):
        return deepcopy(self)

    def make_action(self, their_action):
        self.their_history.append(their_action)
        # Update the list of possibilities for each agent
        for agent_probability in self.agent_probabilities:
            agent_probability.update(their_action)
        for agent in self.agent_probabilities:
            agent.predict_defect_percent(self)

    def __repr__(self):
        return f"NathanModelState({self.my_history}, {self.their_history})"

class AgentProbabilities:
    def __init__(self, agent, possibility, simulations=10):
        self.agent = agent
        self.possibility = possibility
        self.percent_defect_prediction = None
        self.simulations = simulations
        self.state = agent.initial_state
        self.defection_state = None
        self.cooperation_state = None

    def predict_defect_percent(self, model_state):
        defection_count = 0
        self.defection_state = None
        self.cooperation_state = None
        for _ in range(self.simulations):
            cloned_state = deepcopy(self.state)
            action, new_state = self.agent.make_decision(model_state.my_history, cloned_state)
            defection_count += 1 if action == Decision.DEFECT else 0
            if action == Decision.DEFECT:
                self.defection_state = new_state
            else:
                self.cooperation_state = new_state
        if self.defection_state is None:
            self.defection_state = self.cooperation_state
        if self.cooperation_state is None:
            self.cooperation_state = self.defection_state

        self.percent_defect_prediction = defection_count / self.simulations

    def update(self, next_action):
        # update probability of agent being this agent
        if next_action == Decision.DEFECT:
            self.possibility *= 1 - ((1 - self.percent_defect_prediction) / 2)
            self.state = self.defection_state
        else:
            self.possibility *= 1 - (self.percent_defect_prediction / 2)
            self.state = self.cooperation_state
        self.possibility = np.clip(self.possibility, 0.001, 1)

    def __repr__(self):
        return f"AgentProbabilities({self.agent.get_name()}, {self.possibility})"
class NathanMarkov(Agent):
    @property
    def initial_state(self) -> AgentState:
        opponents = [TitForTat(), Mac(), Cynic(), Random(random_seed=1), Rube(), Troll(), Binomial(),
                     PatternMatcher(), IForgiveYou(), AdvancedPredict(), EricTheEvil1(), EricTheEvil2(), EricTheEvil3(),
                     TitForTwoTats(), GrimTrigger(), Stephanie(), TribalPolitician(), TribalCultist(),
                     TribalCheater(), Konstantin()]

        return NathanModelState([AgentProbabilities(agent, 1.0) for agent in opponents])

    def simulate(self, model_state, action, depth):
        if depth == 0:
            return 0

        simulated_state = model_state.clone()
        simulated_state.my_history.append(action)

        weighted_scores = []

        for agent_probability in simulated_state.agent_probabilities:
            agent = agent_probability.agent
            agent_action = agent.make_decision(simulated_state.their_history, agent_probability.state)[0]
            simulated_state.their_history.append(agent_action)
            my_action = self.markov_tree_search(deepcopy(simulated_state), depth - 1)

            score = payoff(my_action, agent_action)
            weighted_scores.append((agent_probability.possibility, score))

            simulated_state.their_history.pop()

        return sum([score * possibility for possibility, score in weighted_scores])

    def markov_tree_search(self, model_state, depth):
        defect_score = self.simulate(model_state, Decision.DEFECT, depth)
        cooperate_score = self.simulate(model_state, Decision.COOPERATE, depth)
        print(defect_score, cooperate_score, depth, model_state)
        if defect_score > cooperate_score:
            return Decision.DEFECT
        else:
            return Decision.COOPERATE

    def make_decision(self, other_agents_decisions: Tuple[Decision, ...], previous_state: AgentState) -> Tuple[
        Decision, AgentState]:

        model_state = previous_state
        if len(other_agents_decisions) == 0:
            model_state.my_history.append(Decision.COOPERATE)
            return Decision.COOPERATE, model_state

        model_state.make_action(other_agents_decisions[0])

        model_state.their_history.append(other_agents_decisions[0])

        my_action = self.markov_tree_search(model_state, 5)

        model_state.make_action(my_action)

        model_state.my_history.append(my_action)
        return my_action, model_state



if __name__ == '__main__':
    nathan = NathanMarkov()
    state = nathan.initial_state
    nathan.make_decision([], state)
    nathan.make_decision((Decision.DEFECT,), state)
    nathan.make_decision((Decision.DEFECT, Decision.DEFECT,), state)
    nathan.make_decision((Decision.DEFECT, Decision.DEFECT, Decision.DEFECT,), state)
    nathan.make_decision((Decision.DEFECT, Decision.DEFECT, Decision.DEFECT, Decision.DEFECT,), state)
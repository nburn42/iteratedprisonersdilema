from __future__ import annotations

import abc
import enum
import itertools
import json
import random
from typing import Any, Optional, Tuple

import attrs
import numpy as np
from attr import attr


@enum.unique
class Decision(enum.Enum):
    COOPERATE = 0
    DEFECT = 1


AgentState = Any


class Agent(abc.ABC):
    def get_name(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        """Make a new decision in an iterated prisoner's dilemma.

        Args:
          other_agents_decisions: List of decisions the other agent has made in
            your previous interactions with them, ordered from most recent to least
            recent.
          previous_state: Some object that encodes the previous internal state of
            this agent.
        Return:
          The next decision your agent makes in the iterated prisoner's dilemma, and
          the current internal state of this agent (to be passed on to subsequent
          calls to `make_decision`).
        """
        ...

    @property
    @abc.abstractmethod
    def initial_state(self) -> AgentState:
        """The initial internal state of this agent."""
        ...


def payoff(
        your_decision: Decision,
        other_decision: Decision,
) -> int:
    if your_decision == Decision.COOPERATE:
        if other_decision == Decision.COOPERATE:
            return 3
        if other_decision == Decision.DEFECT:
            return 0
    if your_decision == Decision.DEFECT:
        if other_decision == Decision.COOPERATE:
            return 5
        if other_decision == Decision.DEFECT:
            return 1


def play_iterated_prisoners_dilemma(
        agent_1: Agent,
        agent_2: Agent,
        expeted_number_of_interactions: float = 200.0,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    if rng is None:
        rng = np.random.default_rng()
    num_interactions = rng.poisson(expeted_number_of_interactions)
    score_1 = 0
    score_2 = 0
    agent_1_state = agent_1.initial_state
    agent_2_state = agent_2.initial_state
    agent_1_decisions = ()
    agent_2_decisions = ()
    for interaction in range(num_interactions):
        agent_1_decision, agent_1_state = agent_1.make_decision(
            other_agents_decisions=agent_2_decisions,
            previous_state=agent_1_state,
        )

        agent_2_decision, agent_2_state = agent_2.make_decision(
            other_agents_decisions=agent_1_decisions,
            previous_state=agent_2_state,
        )

        agent_1_decisions = (agent_1_decision,) + agent_1_decisions
        agent_2_decisions = (agent_2_decision,) + agent_2_decisions

        score_1 += payoff(
            your_decision=agent_1_decision,
            other_decision=agent_2_decision,
        )
        score_2 += payoff(
            your_decision=agent_2_decision,
            other_decision=agent_1_decision,
        )
    return round(score_1 / num_interactions, 2), round(score_2 / num_interactions, 2)


@attrs.frozen
class Frame:
    agent_names: list[str]
    agent_scores: list[int]
    current_pair: tuple[str, str]
    current_decisions: list[tuple[Decision, Decision]]
    cumulative_decision_pairs: dict[tuple[Decision, Decision], int]

    def to_json(self):
        data = {
            "agent_names": self.agent_names,
            "agent_scores": self.agent_scores,
            "current_pair": self.current_pair,
            "current_decisions": self.current_decisions,
            "cumulative_decision_pairs": self.cumulative_decision_pairs,
        }
        data["current_decisions"] = [
            (decision[0].name, decision[1].name)
            for decision in data["current_decisions"]
        ]
        data["cumulative_decision_pairs"] = {
            str(key): value
            for key, value in data["cumulative_decision_pairs"].items()
        }
        return json.dumps(data)


    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        data["current_decisions"] = [
            (Decision[decision[0]], Decision[decision[1]])
            for decision in data["current_decisions"]
        ]
        data["cumulative_decision_pairs"] = {
            (Decision[key[0]], Decision[key[1]]): value
            for key, value in data["cumulative_decision_pairs"].items()
        }
        return cls(**data)


    def update_with_decision_pair(self, agent_1_decision: Decision, agent_2_decision: Decision):
        agent_1_payoff = payoff(
            your_decision=agent_1_decision,
            other_decision=agent_2_decision,
        )
        agent_2_payoff = payoff(
            your_decision=agent_2_decision,
            other_decision=agent_1_decision,
        )
        agent_1_idx = self.agent_names.index(self.current_pair[0])
        agent_2_idx = self.agent_names.index(self.current_pair[1])
        new_scores = self.agent_scores.copy()
        new_scores[agent_1_idx] += agent_1_payoff
        new_scores[agent_2_idx] += agent_2_payoff
        new_cumulative_decision_pairs = self.cumulative_decision_pairs.copy()
        new_cumulative_decision_pairs[agent_1_decision, agent_2_decision] += 1
        return Frame(
            agent_names=self.agent_names,
            agent_scores=new_scores,
            current_pair=self.current_pair,
            current_decisions=[(agent_1_decision, agent_2_decision)] + self.current_decisions,
            cumulative_decision_pairs=new_cumulative_decision_pairs,
        )



@attrs.frozen
class FaceoffResult:
    agent_1_decisions: list[Decision]
    agent_2_decisions: list[Decision]
    agent_1_payoffs: list[int]
    agent_2_payoffs: list[int]


def faceoff_iterated_prisoners_dilemma(
        agent_1: Agent,
        agent_2: Agent,
        expeted_number_of_interactions: float = 200.0,
        rng: Optional[np.random.Generator] = None,
) -> FaceoffResult:
    if rng is None:
        rng = np.random.default_rng()
    num_interactions = rng.poisson(expeted_number_of_interactions)
    agent_1_payoffs = ()
    agent_2_payoffs = ()
    agent_1_state = agent_1.initial_state
    agent_2_state = agent_2.initial_state
    agent_1_decisions = ()
    agent_2_decisions = ()
    for interaction in range(num_interactions):
        agent_1_decision, agent_1_state = agent_1.make_decision(
            other_agents_decisions=agent_2_decisions,
            previous_state=agent_1_state,
        )

        agent_2_decision, agent_2_state = agent_2.make_decision(
            other_agents_decisions=agent_1_decisions,
            previous_state=agent_2_state,
        )

        agent_1_decisions = (agent_1_decision,) + agent_1_decisions
        agent_2_decisions = (agent_2_decision,) + agent_2_decisions

        agent_1_payoffs = (payoff(
            your_decision=agent_1_decision,
            other_decision=agent_2_decision,
        ),) + agent_1_payoffs
        agent_2_payoffs = (payoff(
            your_decision=agent_2_decision,
            other_decision=agent_1_decision,
        ),) + agent_2_payoffs
    return FaceoffResult(
        agent_1_decisions=agent_1_decisions,
        agent_2_decisions=agent_2_decisions,
        agent_1_payoffs=agent_1_payoffs,
        agent_2_payoffs=agent_2_payoffs,
    )


def faceoff_result_to_frames(
        faceoff_result: FaceoffResult,
        agent_names: list[str],
        agent_scores: list[int],
        current_agents: tuple[Agent],
) -> list[Frame]:
    frames = [
        Frame(
            agent_names=agent_names,
            agent_scores=agent_scores,
            current_pair=tuple(agent.__class__.__name__ for agent in current_agents),
            current_decisions=[],
            cumulative_decision_pairs={(d1, d2): 0 for d1, d2 in itertools.product(Decision, repeat=2)}
        )
    ]
    for agent_1_decision, agent_2_decision in zip(faceoff_result.agent_1_decisions[::-1],
                                                  faceoff_result.agent_2_decisions[::-1]):
        frames.append(
            frames[-1].update_with_decision_pair(
                agent_1_decision=agent_1_decision,
                agent_2_decision=agent_2_decision,
            )
        )
    return frames


def contest_to_frames(
        pair_results: list[tuple[tuple[Agent, Agent], FaceoffResult]],
):
    agent_names = sorted(set(itertools.chain(*[[a.__class__.__name__ for a in pair] for pair, _ in pair_results])))
    agent_scores = len(agent_names) * [0]
    frames = []
    shuffled_results = pair_results.copy()
    random.shuffle(shuffled_results)
    for pair_result in shuffled_results:
        pair, result = pair_result
        frames += faceoff_result_to_frames(
            faceoff_result=result,
            agent_names=agent_names,
            agent_scores=agent_scores,
            current_agents=pair,
        )
        agent_scores = frames[-1].agent_scores
    return frames

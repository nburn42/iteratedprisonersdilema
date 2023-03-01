
from __future__ import annotations

import abc
import enum
from typing import Any, Optional, Tuple

import numpy as np


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


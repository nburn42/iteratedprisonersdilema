from typing import Tuple

from agents import Random, TitForTwoTats
from utils import AgentState, Decision, Agent, play_iterated_prisoners_dilemma


class Jennifer(Agent):
    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        if not other_agents_decisions:
            return Decision.COOPERATE, None
        return other_agents_decisions[0], None





if __name__ == '__main__':
    print(play_iterated_prisoners_dilemma(
        agent_1=Jennifer(),
        agent_2=TitForTwoTats(random_seed=1),
    ))

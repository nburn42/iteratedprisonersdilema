import random
from typing import Tuple

import attrs
import numpy as np

from utils import Decision, AgentState, Agent


class TitForTat(Agent):
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


@attrs.frozen
class Random(Agent):
    random_seed: int

    @property
    def initial_state(self) -> AgentState:
        return np.random.default_rng(self.random_seed)

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        decision = previous_state.choice(Decision)
        return decision, previous_state


class Mac(Agent):
    @property
    def initial_state(self) -> AgentState:
        return [0, 2]  # serial_defections, cynicism

    def cynicism(agent, state):
        hope = False
        if state[0] == state[1] ** 2:
            hope = True
        if state[0] == state[1] ** 2 + 1:
            state[0] = 0
            state[1] += 1
            hope = True
        return hope, state

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        if other_agents_decisions:

            if other_agents_decisions[0] == Decision.DEFECT:
                previous_state[0] += 1
            else:
                previous_state[0] = 0

            play, new_state = self.cynicism(previous_state)
            if play:
                return Decision.COOPERATE, new_state
            else:
                # previous_state[1] += 1
                return other_agents_decisions[0], new_state
        else:
            return Decision.COOPERATE, previous_state


class Cynic(Agent):
    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        return Decision.DEFECT, None


class Rube(Agent):
    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        return Decision.COOPERATE, None


class Troll(Agent):
    @property
    def initial_state(self) -> AgentState:
        return [0, 0]  # [troll, grim trigger]

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        if 1 in previous_state:
            return Decision.DEFECT, previous_state

        num_rounds = len(other_agents_decisions)

        if num_rounds < 2:
            return Decision.DEFECT, previous_state

        elif num_rounds < 3 and (Decision.DEFECT not in other_agents_decisions):
            previous_state[0] = 1  # start trolling
            return Decision.DEFECT, previous_state

        elif num_rounds < 4:
            return Decision.COOPERATE, previous_state
        else:
            if other_agents_decisions[0] == Decision.COOPERATE:
                return Decision.COOPERATE, previous_state
            else:
                previous_state[1] = 1  # grim trigger
                return Decision.DEFECT, previous_state


class TitForTwoTats(Agent):
    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds < 2:
            return Decision.COOPERATE, None

        if other_agents_decisions[0] == Decision.DEFECT and other_agents_decisions[1] == Decision.DEFECT:
            return Decision.DEFECT, None
        return Decision.COOPERATE, None


class GrimTrigger(Agent):
    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        if Decision.DEFECT in other_agents_decisions:
            return Decision.DEFECT, None
        else:
            return Decision.COOPERATE, None

class EricTheEvil1(Agent):
  @property
  def initial_state(self) -> AgentState:
    return None

  def make_decision(
      self,
      other_agents_decisions: Tuple[Decision, ...],
      previous_state: AgentState,
  ) -> Tuple[Decision, AgentState]:

    d = sum(int(d==Decision.DEFECT) for d in other_agents_decisions)
    t = len(other_agents_decisions) or 1
    r = random.random()
    return [Decision.COOPERATE, Decision.DEFECT][r < d/t], None


class EricTheEvil2(Agent):
   @property
   def initial_state(self) -> AgentState:
     return np.array([0,0])

   def make_decision(
       self,
       other_agents_decisions: Tuple[Decision, ...],
       previous_state: AgentState
   ) -> Tuple[Decision, AgentState]:

     v0 = [np.array([-3, -5]), np.array([0, -1])]
     if len(other_agents_decisions):
       d = other_agents_decisions[-1]
       previous_state += v0[d != Decision.COOPERATE]

     u = previous_state + v0[random.random() < .5]
     d = Decision.COOPERATE if u[0] < u[1] else Decision.DEFECT
     return d, previous_state


class EricTheEvil3(Agent):
    @property
    def initial_state(self) -> AgentState:
        return []

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState
    ) -> Tuple[Decision, AgentState]:

        if len(other_agents_decisions):
            previous_state[-1].append(other_agents_decisions[-1])

        payoff = [[3, 0], [5, 1]]

        # compute (un-normalized) expected values
        c, d = 0, 0
        for (me, them) in previous_state:
            me, them = me.value, them.value
            [c, d][me] += payoff[me][them]

        res = Decision.COOPERATE if c >= d else Decision.DEFECT
        previous_state.append([res])
        return res, previous_state

class AdvancedPredict(Agent):
    """By Emmett
    Tallies count of each of opponents's previous decisions, weighing recent
    decisions more than older ones.
    """

    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:
        cooperateVal: float = 0
        defectVal: float = 0
        initWeight: int = 1  # weight starts at this value
        weightLoss: float = 0.9  # next decision has this much less weight

        weight: float = initWeight

        for i in other_agents_decisions:
            if i == Decision.COOPERATE:
                cooperateVal += weight
            else:
                defectVal += weight
            weight *= weightLoss  # reduce weight of next iteration

        if cooperateVal >= defectVal:
            return Decision.COOPERATE, None
        else:
            return Decision.DEFECT, None


class Stephanie(Agent):
    @property
    def initial_state(self) -> AgentState:
        return 0

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        n = 10  # How long s

        if len(other_agents_decisions) <= n:  # Cooperate for first n rounds
            return Decision.COOPERATE, previous_state

        if previous_state > 0 and other_agents_decisions[
            0] == Decision.DEFECT:  # If I've tried to exploit in the last 5 rounds and you defected last round, cooperate up to n times to try to get forgived
            return Decision.COOPERATE, previous_state - 1

        myBad = n  # Parameter for apology rate

        lastN = 1 - (sum(int(d == Decision.DEFECT) for d in other_agents_decisions[:n]) / min(
            len(other_agents_decisions), n))
        # allTrials = 1-(sum(int(d==Decision.DEFECT) for d in other_agents_decisions)/len(other_agents_decisions))

        if len(other_agents_decisions) >= n + 1 and lastN == 1:  # If they've cooperated for the last n+1 times, try to exploit
            return Decision.DEFECT, myBad  # In case we need forgiveness

        # Otherwise, cooperate if 75% of the last n trials are cooperate
        if lastN >= 0.75:
            return Decision.COOPERATE, previous_state
        else:
            return Decision.DEFECT, previous_state


class PatternMatcher(Agent):
    """ By Matt
        Looks at how the opponent has responded to its last 3 plays historically
        and selects the action more likely to get a cooperate (with a margin to
        catch random)
    """

    @property
    def initial_state(self) -> AgentState:
        return []

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds < 5:
            if not other_agents_decisions:
                my_decision = Decision.COOPERATE
                return my_decision, [my_decision]
            else:
                my_decision = other_agents_decisions[0]
        else:
            prob_defect = {}
            for my_dec in [Decision.DEFECT, Decision.COOPERATE]:
                # Use last 2 and test what I do next
                pattern = previous_state[1:3]
                pattern = [my_dec] + pattern
                outcomes = []
                for i in range(1, len(previous_state) - len(pattern)):
                    if previous_state[i:i + len(pattern)] == pattern:
                        outcomes.append(other_agents_decisions[i - 1])
                defects = sum([d == Decision.DEFECT for d in outcomes])
                trials = len(outcomes)

                # Add ballast
                trials += 2
                defects += 2 * sum([d == Decision.DEFECT for d in other_agents_decisions]) / len(other_agents_decisions)

                prob_defect[my_dec] = defects / trials

            margin = 0.05
            if prob_defect[Decision.DEFECT] > (prob_defect[Decision.COOPERATE] + margin):
                my_decision = Decision.COOPERATE
            else:
                my_decision = Decision.DEFECT
        return my_decision, [my_decision] + previous_state


class IForgiveYou(Agent):
    """ By Matt
        Cooperates until you defect twice. Then you're dead to it.
    """

    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds < 2:
            return Decision.COOPERATE, None

        defections = sum([d == Decision.DEFECT for d in other_agents_decisions])
        if defections > 2:
            return Decision.DEFECT, None
        else:
            return Decision.COOPERATE, None


import operator


@attrs.frozen
class BinomialState:
    my_decisions: list[Decision]
    decision_matrix: np.ndarray

    def add_decision(
            self,
            my_decision: Decision,
    ):
        return BinomialState(
            my_decisions=self.my_decisions + [my_decision],
            decision_matrix=self.decision_matrix,
        )

    def update_decision_matrix(
            self,
            other_decision: Decision,
    ):
        decision_matrix = self.decision_matrix.copy()
        if len(self.my_decisions) > 1:
            row_idx = self.my_decisions[-2].value
            col_idx = other_decision.value
            decision_matrix[row_idx, col_idx] += 1
        return BinomialState(
            my_decisions=self.my_decisions,
            decision_matrix=decision_matrix,
        )


class Binomial(Agent):
    @property
    def initial_state(self) -> AgentState:
        return BinomialState(
            my_decisions=[],
            decision_matrix=np.zeros((2, 2), dtype=int),
        )

    @staticmethod
    def other_is_random(decision_matrix: np.ndarray) -> bool:
        reaction_counts = np.sum(decision_matrix, axis=1)
        if np.any(reaction_counts < 2):
            return False
        conditional_defection_probs = decision_matrix[:, 1] / reaction_counts
        std_devs = np.sqrt(conditional_defection_probs * (1 - conditional_defection_probs) / (reaction_counts - 1))
        return np.logical_and(
            conditional_defection_probs + 2 * std_devs < 1,
            np.logical_and(
                conditional_defection_probs - 2 * std_devs > 0,
                np.abs(conditional_defection_probs - 0.5) < 2 * std_devs,
            ),
        ).all()

    def make_decision(self, other_agents_decisions: Tuple[Decision, ...], previous_state: AgentState) -> Tuple[
        Decision, AgentState]:
        if not other_agents_decisions:
            # Cooperate on first round.
            my_decision = Decision.COOPERATE
            current_state = previous_state.add_decision(my_decision)
            return my_decision, current_state
        current_state = previous_state.update_decision_matrix(other_agents_decisions[0])
        if self.other_is_random(current_state.decision_matrix):
            my_decision = Decision.DEFECT
        else:
            my_decision = other_agents_decisions[0]
        current_state = current_state.add_decision(my_decision)
        return my_decision, current_state


class TribalPolitician(Agent):
    """ By Ryan
        Tries to identify other members of its tribe - then defects against them while going tit for tat against others
    """

    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return Decision.COOPERATE, None
        if num_rounds == 1:
            return Decision.DEFECT, None
        if num_rounds == 2:
            return Decision.COOPERATE, None
        if num_rounds == 3:
            return Decision.COOPERATE, None

        othertribal = 0
        meetingofkings = 0
        if num_rounds > 3:
            if other_agents_decisions[-4] == Decision.COOPERATE and other_agents_decisions[-3] == Decision.COOPERATE and \
                    other_agents_decisions[-2] == Decision.DEFECT and other_agents_decisions[-1] == Decision.COOPERATE:
                othertribal = 1

        if num_rounds == 4 and othertribal:
            return Decision.DEFECT, None
        elif num_rounds == 4:
            return other_agents_decisions[0], None

        if othertribal and other_agents_decisions[-5] == Decision.DEFECT:
            meetingofkings = 1

        if othertribal and meetingofkings:
            return Decision.COOPERATE, None
        elif othertribal:
            return Decision.DEFECT, None
        else:
            return other_agents_decisions[0], None


class TribalCultist(Agent):
    """ By Ryan
        Tries to identify other members of its tribe - then cooperates with them while defecting against anyone else
    """

    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return Decision.COOPERATE, None
        if num_rounds == 1:
            return Decision.DEFECT, None
        if num_rounds == 2:
            return Decision.COOPERATE, None
        if num_rounds == 3:
            return Decision.COOPERATE, None

        if num_rounds > 3:
            if other_agents_decisions[-4] == Decision.COOPERATE and other_agents_decisions[-3] == Decision.COOPERATE and \
                    other_agents_decisions[-2] == Decision.DEFECT and other_agents_decisions[-1] == Decision.COOPERATE:
                return Decision.COOPERATE, None

            return Decision.DEFECT, None


class TribalCheater(Agent):
    """ By Ryan
        Tries to identify other members of its tribe - then shows deference to politicians while going tit-for-tat against others (with an added chance of defecting anyways)
    """

    @property
    def initial_state(self) -> AgentState:
        return None

    def make_decision(
            self,
            other_agents_decisions: Tuple[Decision, ...],
            previous_state: AgentState,
    ) -> Tuple[Decision, AgentState]:

        num_rounds = len(other_agents_decisions)

        if num_rounds == 0:
            return Decision.COOPERATE, None
        if num_rounds == 1:
            return Decision.DEFECT, None
        if num_rounds == 2:
            return Decision.COOPERATE, None
        if num_rounds == 3:
            return Decision.COOPERATE, None

        othertribal = 0
        if num_rounds > 3:
            if other_agents_decisions[-4] == Decision.COOPERATE and other_agents_decisions[-3] == Decision.COOPERATE and \
                    other_agents_decisions[-2] == Decision.DEFECT and other_agents_decisions[-1] == Decision.COOPERATE:
                othertribal = 1

        if num_rounds > 3 and othertribal:
            return Decision.COOPERATE, None
        else:
            if random.random() < .1:
                return Decision.DEFECT, None
            else:
                return other_agents_decisions[0], None

class Konstantin(Agent):
  """ By Konstantin, Implementation by Adam
      Description: Cooperates on the first move, and defects if the opponent has defects on any of the previous three moves, else cooperates.
  """
  @property
  def initial_state(self) -> AgentState:
    return None

  def make_decision(
      self,
      other_agents_decisions: Tuple[Decision, ...],
      previous_state: AgentState,
  ) -> Tuple[Decision, AgentState]:
    if not other_agents_decisions: #if it's the first round
      return Decision.COOPERATE, None
    elif any([d == Decision.DEFECT for d in other_agents_decisions[:3]]): #if villain has defected in the last 3 rounds
      return Decision.DEFECT, None
    else: #otherwise
      return Decision.COOPERATE, None
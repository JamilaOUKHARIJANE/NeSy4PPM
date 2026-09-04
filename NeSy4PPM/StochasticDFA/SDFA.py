from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import FrozenSet, Optional
from collections.abc import Iterable

State = int
Activity = str

@dataclass(frozen=True)
class StochasticDFA:
    """
    Stochastic DFA S = (Q, A, delta, delta_p, q0).

    delta maps (state, activity) -> next_state.
    delta_p maps (state, activity) -> transition probability.
    """

    Q: FrozenSet[State]
    A: FrozenSet[Activity]
    delta: dict[tuple[State, Activity], State]
    delta_p: dict[tuple[State, Activity], float]
    q0: State
    tau: dict[State, float] | None = None

    @classmethod
    def from_sdfa_file(cls, sdfa_file: str | Path) -> "StochasticDFA":
        data = json.loads(Path(sdfa_file).read_text(encoding="utf-8"))
        q0 = int(data["initialState"])
        states = {q0}
        alphabet = set()
        delta = {}
        delta_p = {}
        tau = {
            int(state): float(Fraction(str(probability)))
            for state, probability in data.get("terminationProbabilities", {}).items()
        }

        for transition in data.get("transitions", []):
            source = int(transition["from"])
            target = int(transition["to"])
            activity = str(transition["label"])
            probability = float(Fraction(str(transition["prob"])))
            key = (source, activity)

            states.update((source, target))
            alphabet.add(activity)
            delta[key] = target
            delta_p[key] = probability

        return cls(
            Q=frozenset(states),
            A=frozenset(alphabet),
            delta=delta,
            delta_p=delta_p,
            q0=q0,
            tau=tau or None,
        )


    def next_state(self, state: State, activity: Activity) -> Optional[State]:
        return self.delta.get((state, activity), None)

    def probability(self, state: State, activity: Activity) -> float:
        return self.delta_p.get((state, activity), 0.0)

    def end_probability(self, state: State) -> float:
        if self.tau is not None:
            return self.tau.get(state, 0.0)
        outgoing_probability = sum(probability for (source, _), probability in self.delta_p.items() if source == state)
        return max(0.0, 1.0 - outgoing_probability)

    def replay(self, trace, start_state: Optional[State] = None) -> Optional[State]:
        state = self.q0 if start_state is None else start_state

        for activity in trace:
            next_state = self.next_state(state, activity)
            if next_state is None:
                next_state= state #return None
            state = next_state
        return state

    def strict_replay(self, trace, start_state: Optional[State] = None) -> Optional[State]:
        """
        Strictly replay activities on the SDFA.

        Returns the reached state, or None when an activity does not
        correspond to an admissible transition.
        """
        state = self.q0 if start_state is None else start_state

        for activity in trace:
            next_state = self.next_state(state, activity)
            if next_state is None or self.probability(state, activity) == 0.0:
                return None
            state = next_state
        return state

    def termination_distance(self, state: State) -> float:
        """
        Compute D_tau(state): the shortest path length to a state q'
        with tau(q') > 0. Returns float("inf") when no such state is reachable.
        """
        if self.end_probability(state) > 0.0:
            return 0

        visited = {state}
        queue = deque([(state, 0)])

        while queue:
            current_state, distance = queue.popleft()
            for (source, _), target in self.delta.items():
                if source != current_state or target in visited:
                    continue
                if self.end_probability(target) > 0.0:
                    return distance + 1
                visited.add(target)
                queue.append((target, distance + 1))

        return float("inf")

    def termination_probability(self, state: State, horizon: int) -> float:
        """
        Compute G_tau^R(state): the maximum product probability of any feasible
        continuation of total length at most horizon that ends with termination.

        The terminating symbol is counted in the horizon, so a path with m
        activities is considered only when m + 1 <= horizon.
        """
        if horizon <= 0:
            return 0.0

        best_termination_probability = self.end_probability(state)
        current_layer = {state: 1.0}

        for depth in range(horizon - 1):
            next_layer = {}
            for current_state, path_probability in current_layer.items():
                for (source, activity), target in self.delta.items():
                    if source != current_state:
                        continue

                    transition_probability = self.delta_p[(source, activity)]
                    if transition_probability <= 0.0:
                        continue

                    candidate_probability = path_probability * transition_probability
                    if candidate_probability > next_layer.get(target, 0.0):
                        next_layer[target] = candidate_probability

                    termination_probability = (
                        candidate_probability * self.end_probability(target)
                    )
                    if termination_probability > best_termination_probability:
                        best_termination_probability = termination_probability

            if not next_layer:
                break
            current_layer = next_layer

        return best_termination_probability

    def is_compliant_trace(self,trace: Iterable[Activity],epsilon: float = 0) -> bool:
        """
        Return True iff the complete trace can be replayed and terminates
        in a state where termination is allowed.
        """
        final_state = self.strict_replay(trace)
        if final_state is None:
            return False

        return self.end_probability(final_state) > epsilon

    def suffix_compliance( self, prefix: Iterable[Activity], predicted_suffix: Iterable[Activity], epsilon: float = 0, termination=False) -> Optional[bool]:
        """
        Evaluate the compliance of a predicted suffix.

        Returns:
            True:  prefix and suffix form a compliant complete trace or suffix is compliant from starting state.
            False: the suffix is infeasible or cannot terminate.
        """
        prefix_state = self.replay(prefix)
        final_state = self.strict_replay(predicted_suffix,start_state=prefix_state)
        if final_state is None:
            return False
        return self.end_probability(final_state) > epsilon if termination else True

    def compute_compliance_metrics(self,predictions: list[tuple[list[Activity], list[Activity]]]) -> dict[str, float]:
        """
        predictions contains (prefix, predicted_suffix) pairs.
        """
        feasibility_results = [ self.suffix_compliance(prefix, suffix) for prefix, suffix in predictions]
        feasibility_Termination_results = [ self.suffix_compliance(prefix, suffix, termination=True) for prefix, suffix in predictions]

        total = len(feasibility_results)
        if total == 0:
            return {
                "feasibility_rate": 0.0,
                "termination_rate": 0.0,
            }

        terminated_count = sum(result is True for result in feasibility_Termination_results)
        feasible_count = sum(result is True for result in feasibility_results)

        return {
            "feasibility_rate": feasible_count / total,
            "termination_rate": terminated_count / total,
        }

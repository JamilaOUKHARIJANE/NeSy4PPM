from enum import Enum
from functools import reduce

from pythomata.core import DFA
from typing import TypeVar, Sequence

StateType = TypeVar("StateType")
SymbolType = TypeVar("SymbolType")


class TruthValue(Enum):
    PERM_SAT = 1
    POSS_SAT = 2
    POSS_VIOL = 3
    PERM_VIOL = 4


#Translates a prefix from a list of strings to a dictionary that can be used with the FiniteAutomaton class from pythomata/core.py
@staticmethod
def prefix_to_word(prefix: list[str], activityToEncoding: dict[str,str]) -> Sequence[SymbolType]:
    word = [{}]  #The automaton from ltl2dfa seems to always have an initial state with a single outgoing arc labeled true
    for activity in prefix:
        if activity in activityToEncoding:
            word.append({activityToEncoding[activity]:True}) #Ocurrence of an activity is represented by setting the corresponding proposition to true, other propositions remain false by default
        else:
            word.append({}) #Activities that are not present in the decl model are represented as an empty dictionary (removing the empty dictionary would affect the processing of chain type constraints)
    return word



#Returns the state of the automaton after processing a given word
#Based on FiniteAutomaton.accepts method from pythomata/core.py
@staticmethod
def get_state_for_prefix(aut: DFA, word: Sequence[SymbolType]) -> StateType:
    current_states = {aut.initial_state} #Reset the automaton to the initial state
    for symbol in word: #Find the automaton state after replaying the input word
            current_states = reduce(
                set.union,  # type: ignore
                map(lambda x: aut.get_successors(x, symbol), current_states),
                set(),
            )

    return list(current_states)[0] #The input is a DFA, so there must always be exactly one current state

#Finds the truth value of a given state in the given automaton
#Assumes that the given automaton is minimized
@staticmethod
def get_state_truth_value(aut: DFA, state: StateType, activityEncodings: list[str]) -> TruthValue:
    accepting = aut.is_accepting(state)
    for activityEncoding in activityEncodings:
        #If it is possible to exit the given state with at least one possible activity then it is not a trap-state and therefore the truth value is temporary
        for successor in aut.get_successors(state, {activityEncoding: True}):
            if not(state == successor):
                return TruthValue.POSS_SAT if accepting else TruthValue.POSS_VIOL
        for successor in aut.get_successors(state, {}): #Checking the successor for activities not present in the Declare model (needed for chain type constraints)
            if not(state == successor):
                return TruthValue.POSS_SAT if accepting else TruthValue.POSS_VIOL
    return TruthValue.PERM_SAT if accepting else TruthValue.PERM_VIOL #Reaching here means that the given state is a trap-state and therefore the truth value is permanent

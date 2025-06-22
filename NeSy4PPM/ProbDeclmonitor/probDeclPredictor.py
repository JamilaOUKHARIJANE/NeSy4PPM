import re
import itertools
from enum import Enum

import numpy as np

from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessModels.DeclareModel import DeclareModelTemplate
from Declare4Py.ProcessModels.LTLModel import LTLModel

from logaut import ltl2dfa

from scipy.optimize import linprog

from NeSy4PPM.ProbDeclmonitor  import ltlUtils
from NeSy4PPM.ProbDeclmonitor import autUtils
from NeSy4PPM.ProbDeclmonitor.autUtils import TruthValue


class AggregationMethod(Enum):
    SUM = 1
    MAX = 2
    AVG = 3
    MIN = 4



#Class for the probDeclare predictor
class ProbDeclarePredictor:
    def __init__(self):
        super().__init__()
        self.int_char_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'l'} #Taken from Declare4Py.Utils.utils.parse_activity(act: str)
        self.activityToEncoding = {} #Activity Encodings, used internally to avoid issues with special characters in activity names
        self.constraintFormulas = [] #Ltl formula of each constraint in the same order as they appear in the input model (with encoded activities)
        self.formulaToProbability = {} #For looking up probabilities based on constraint formula
        self.scenarios = [] #One tuple per each constraint scenario, tuples consist of 1,0 values where 1 means positive constraint and 0 means negated constraint
        self.inconsistentScenarios = [] #Logically inconsistent scenarios, could probably remove this list and use 'not in scenarioToDfa.keys()' instead
        self.scenarioToDfa = {} #For looking up DFA based on the scenario, contains only consistent scenarios
        self.scenarioToProbability = {} #For looking up the probability of a secanrio
    

    def loadProbDeclModel(self, modelPath: str) -> None: #Reads and processes the input declare model
        #Reading the decl model
        with open(modelPath, "r+") as file:
            for line in file:
                #Assuming <constraint>;<probability>
                splitLine = line.split(";")
                probability = float(splitLine[1].strip())
                constraintStr = splitLine[0].strip()
                
                if DeclareModel.is_constraint_template_definition(constraintStr):
                    #Based on the method Declare4Py.ProcessModels.DeclareModel.parse(self, lines: [str])
                    split = constraintStr.split("[", 1)
                    template_search = re.search(r'(^.+?)(\d*$)', split[0])
                    if template_search is not None:
                        template_str, cardinality = template_search.groups()
                        template = DeclareModelTemplate.get_template_from_string(template_str)
                        if template is not None:
                            activities = split[1].split("]")[0]
                            activities = activities.split(", ")
                            tmp = {"template": template, "activities": activities,
                                    "condition": re.split(r'\s+\|', constraintStr)[1:]}
                            if template.supports_cardinality:
                                tmp['n'] = 1 if not cardinality else int(cardinality)
                                cardinality = tmp['n']
                            
                            #Create activity encoding, if not already created
                            for activity in activities:
                                if activity not in self.activityToEncoding:
                                    activityEncoding = str(len(self.activityToEncoding))
                                    for int_key in self.int_char_map.keys():
                                        activityEncoding = activityEncoding.replace(str(int_key), self.int_char_map[int_key])
                                    self.activityToEncoding[activity] = activityEncoding
                            
                            #Create LTL formula for the constraint
                            formula = ltlUtils.get_constraint_formula(template,
                                                                    self.activityToEncoding[activities[0]],
                                                                    self.activityToEncoding[activities[1]] if template.is_binary else None,
                                                                    cardinality)
                            self.formulaToProbability[formula] = probability
                            self.constraintFormulas.append(formula)
                            
                            print(formula)

        print("Activity Encodings: " + str(self.activityToEncoding))

        print()
        print("======")
        print("Reading decl file done")
        print("======")
        print()


        #Formula for enforcing simple trace semantics (requiring one proposition to hold at every time point, proposition z is intended for activities that are not present in the decl model)
        #self.activityToEncoding[""] = "z" #Used for activities that are not in the decl model
        #simpleTraceFormula = "(G((" + " || ".join(self.activityToEncoding.values()) + ") && " #At least one proposition must always be true
        #acPairs = list(itertools.combinations(self.activityToEncoding.values(),2)) #Creates all possible activity pairs
        #simpleTraceFormula = simpleTraceFormula + "(!(" + ")) && (!( ".join([" && ".join([ac for ac in acPair]) for acPair in acPairs]) + "))))" #At most one proposition must always be true
        #print("Simple trace semantics formula (silently added to all scenarios): " + simpleTraceFormula)

        #Formula for enforcing simple trace semantics (allowing all propositions to be false, should allow processing activities that are not present in the decl model by simply setting all propositions to false)
        if len(self.activityToEncoding) > 1: #Simple trace semantics must be enforced if the declare model has more than one activiy
            acPairs = list(itertools.combinations(self.activityToEncoding.values(),2))
            simpleTraceFormula = "G((!(" + ")) && (!( ".join([" && ".join([ac for ac in acPair]) for acPair in acPairs]) + ")))" #At most one proposition can be true at any point in time
            print("Simple trace semantics formula (silently added to all scenarios): " + simpleTraceFormula)


        #Used for creating the constraint scenarios, 1 - positive constraint, 0 - negated constraint
        self.scenarios = list(itertools.product([1, 0], repeat=len(self.constraintFormulas))) #Scenario with all positive constraints is first, and scenario with all negated constraints is last

        #Creating automata for (and checking logical consistency of) each scenario
        for  scenario in self.scenarios:
            formulaComponents = []
            for index, posneg in enumerate(scenario):
                if posneg == 1:
                    #Add 1 to the scenario name and use the constraint formula as-is
                    formulaComponents.append(self.constraintFormulas[index])
                else:
                    #Add 0 to the scenario name and negate the constraint formula
                    formulaComponents.append("(!" + self.constraintFormulas[index] + ")")
            
            if len(self.activityToEncoding) > 1: #Simple trace semantics must be enforced if the declare model has more than one activiy
                scenarioFormula = " && ".join(formulaComponents) + " && " + simpleTraceFormula #Scenario formula is a conjunction of negated and non-negated constraint formulas + the formula to enforce simple trace semantics
            else:
                scenarioFormula = " && ".join(formulaComponents)
            
            print("===")
            print("Scenario: " + "".join(map(str, scenario)))
            print("Formula: " + scenarioFormula)

            #Parsing the scenario formula
            scenarioModel = LTLModel()
            scenarioModel.to_ltlf2dfa_backend()
            scenarioModel.parse_from_string(scenarioFormula)
            print("Parsed formula: " + str(scenarioModel.parsed_formula))

            #Creating an automaton for the scenario and checking satisfiability
            scenarioDfa = ltl2dfa(scenarioModel.parsed_formula, backend="ltlf2dfa")
            if len(scenarioDfa.accepting_states) == 0:
                print("Satisfiable: False")
                self.inconsistentScenarios.append(scenario) #Name is used in the system of inequalities
            else:
                print("Satisfiable: True")
                scenarioDfa = scenarioDfa.minimize() #Calling minimize seems to be redundant with the ltlf2dfa backend, but keeping the call just in case
                self.scenarioToDfa[scenario] = scenarioDfa #Used for processing the prefix and predicted events
                #print(str(scenarioDfa.to_graphviz()))

        print()
        print("======")
        print("Logical satisfiability checking done")
        print("======")
        print()


        #Creating the system of (in)equalities to calculate scenario probabilities
        lhs_eq_coeficents = [[1] * len(self.scenarios)] #Sum of all scenarios...
        rhs_eq_values = [1.0] #...equals 1

        for formulaIndex, formula in enumerate(self.constraintFormulas):
            lhs_eq_coeficents.append([scenario[formulaIndex] for scenario in self.scenarios]) #Sum of scenarios where a constraint is not negated...
            rhs_eq_values.append(self.formulaToProbability[formula]) #...equals the probability of that constraint
        #for i in range(len(rhs_eq_values)):
        #    print(str(lhs_eq_coeficents[i]) + " = " + str(rhs_eq_values[i]))

        bounds = [] #Tuples of upper and lower bounds for the value of each variable in the system of (in)equalities, where variables represent the probabilities of scenarios
        maxSatProbSum = 0
        maxSatIndex = 0
        for i, scenario in enumerate(self.scenarios):
            if scenario in self.inconsistentScenarios:
                bounds.append((0,0)) #Probability of an inconsistent scenario must be 0
            else:
                bounds.append((0,1)) #Probability of a consistent scenario must be between 0 and 1
                satProbSum = 0 #The system of (in)equalities will be optimized for the syenario where the sum of satisfied constraint probabilities is the highest 
                for j, posneg in enumerate(scenario):
                    if posneg == 1:
                        satProbSum = satProbSum + self.formulaToProbability[self.constraintFormulas[j]]
                    #else:
                        #satProbSum = satProbSum + (1-self.formulaToProbability[self.constraintFormulas[j]])
                print("Scenario " + "".join(map(str, scenario)) + " satProbSum: " + str(satProbSum))
                if satProbSum > maxSatProbSum:
                    maxSatProbSum = satProbSum
                    maxSatIndex = i
        
        c = [[0] * len(self.scenarios)]
        c[0][maxSatIndex] = -1 #Leads to a solution where the scenario at maxSatIndex gets the highest possible probability
        print()

        #c = [[1] * len(self.scenarios)] #Leads to consistent probability values for all scenarios without optimizing for any scenario
        #c[0][1] = -2 #This would instead bias the solution towards assigning a higher probability to the second scenario (while adjusting the probabilities of other scenarios accordingly)
        #print(c)

        #Solving the system of (in)equalities
        res = linprog(c, A_eq=lhs_eq_coeficents, b_eq=rhs_eq_values, bounds=bounds)
        print(res.message)
        if res.success:
            for scenarioIndex, scenarioProbability in enumerate(res.x):
                print("Scenario " + "".join(map(str, self.scenarios[scenarioIndex])) + " probability: " + str(scenarioProbability))
                self.scenarioToProbability[self.scenarios[scenarioIndex]] = scenarioProbability
        else:
            print("No event log can match input constraint probabilities") #For example, the probabilities of Existence[a] and Absence[a] must add up to 1 in every conceivable event log 


        print()
        print("======")
        print("Calculation of scenario probabilities done")
        print("======")
        print()

    
    def processPrefix(self, prefix: list[str], aggregationMethod: AggregationMethod=AggregationMethod.SUM) -> dict[str|bool, np.float64]: #Processes a given prefix based on the currently loaded model

        print()
        print("======")
        print(str(aggregationMethod) + " ranking next activities for prefix " + str(prefix))
        print("======")
        print()

        nextEventScores = {} #Dictionary of next events and their probabilities based on the given prefix and the probDeclare model
        scenarioToPrefixEndState = {} #Dictionary containing the end state of each scenario for the given prefix

        #Finding next possible events (and their probabilities) for a given trace prefix
        word = autUtils.prefix_to_word(prefix, self.activityToEncoding) #Creating the input for DFA based on the given prefix

        #Replay the given prefix and store the resulting state for each scenario automata 
        for scenario, scenarioDfa in self.scenarioToDfa.items():
            prefixEndState = autUtils.get_state_for_prefix(scenarioDfa, word)
            scenarioToPrefixEndState[scenario] = prefixEndState

        #Handling the recommendation for stopping the process execution
        for scenario, prefixEndState in scenarioToPrefixEndState.items():
            #Note that there should always be one scenario that is in either PERM_SAT or POSS_SAT state
            if autUtils.get_state_truth_value(self.scenarioToDfa[scenario], prefixEndState, self.activityToEncoding.values()) is TruthValue.PERM_SAT:
                print("Scenario " + "".join(map(str, scenario)) + " is permanently satisfied. Returning uniform probabilities for all possible futures.")
                prob = 1.0/(len(self.activityToEncoding)+2) #Two is added to account for activities not present in the declare model and stopping
                nextEventScores[False] = prob #Using False for recommending to stop the execution
                nextEventScores[True] = prob #Using True for recommending any activity not present in the declare model (needed for chain type constraints)
                for activity in self.activityToEncoding.keys(): #Setting the score of all other activities to 0.0, the resulting dictionary is the final output
                    nextEventScores[activity] = prob
                break
            if autUtils.get_state_truth_value(self.scenarioToDfa[scenario], prefixEndState, self.activityToEncoding.values()) is TruthValue.POSS_SAT:
                print("Stopping the execution means staying in scenario:")
                print("    " + "".join(map(str, scenario)) + " (probability: " + str(self.scenarioToProbability[scenario]) + ")")
                nextEventScores[False] = self.scenarioToProbability[scenario] #Scores for other potential activities will be added to this dictionary instance

                #Recommendations for the activities that are present in the declare model
                for activity, activityEncoding in self.activityToEncoding.items():
                    tmpProbabilities = []
                    print("The following scenarios would still be possible after executing " + activity + ":")
                    for scenario, scenarioDfa in self.scenarioToDfa.items():
                        successor = list(scenarioDfa.get_successors(scenarioToPrefixEndState[scenario], {activityEncoding: True}))[0] #This is a DFA so there is only one successor
                        if not(autUtils.get_state_truth_value(self.scenarioToDfa[scenario], successor, self.activityToEncoding.values()) is TruthValue.PERM_VIOL):
                            tmpProbabilities.append(self.scenarioToProbability[scenario])
                            print("    " + "".join(map(str, scenario)) + " (probability: " + str(self.scenarioToProbability[scenario]) + ")")
                    nextEventScores[activity] = get_aggregate_score(tmpProbabilities, aggregationMethod)#Aggregate score for the activity according to the selected aggregationMethod
                
                #Recommendation for any activities not present in the declare model
                tmpProbabilities = []
                print("The following scenarios would still be possible after executing an activity not present in the declare model:")
                for scenario, scenarioDfa in self.scenarioToDfa.items():
                    successor = list(scenarioDfa.get_successors(scenarioToPrefixEndState[scenario], {}))[0] #This is a DFA so there is only one successor
                    if not(autUtils.get_state_truth_value(self.scenarioToDfa[scenario], successor, self.activityToEncoding.values()) is TruthValue.PERM_VIOL):
                        tmpProbabilities.append(self.scenarioToProbability[scenario])
                        print("    " + "".join(map(str, scenario)) + " (probability: " + str(self.scenarioToProbability[scenario]) + ")")
                nextEventScores[activity] = get_aggregate_score(tmpProbabilities, aggregationMethod)#Aggregate score for the activity according to the selected aggregationMethod


                break
        print()
        print("======")
        print(str(aggregationMethod) + " ranking of next activities done for prefix " + str(prefix))
        print("======")
        print()

        return nextEventScores
    
@staticmethod
def get_aggregate_score(tmpProbabilities: list[np.float64], aggregationMethod: AggregationMethod) -> np.float64:
    if aggregationMethod is AggregationMethod.SUM: #The score of an event is the sum of the probabilities of scenbarios which are still possible after executing that event
        return np.sum(tmpProbabilities)
    elif aggregationMethod is AggregationMethod.MAX: #The score of an event is the probability of the most likely scenbario which is still possible after executing that event
        return np.max(tmpProbabilities)
    elif aggregationMethod is AggregationMethod.AVG: #The score of an event is the average of the probabilities of scenbarios which are still possible after executing that event
        return np.average(tmpProbabilities)
    elif aggregationMethod is AggregationMethod.MIN: #The score of an event is the probability of the least likely scenbario which is still possible after executing that event
        return np.min(tmpProbabilities)
    else:
        print("Unsupported score aggregation method " + str(aggregationMethod))
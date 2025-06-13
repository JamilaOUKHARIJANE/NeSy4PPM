from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.Utils.Declare import Checkers
from typing import List

from Declare4Py.Utils.Declare.Checkers import CheckerResult, TemplateConstraintChecker


class TraceDeclareAnalyzer(MPDeclareAnalyzer):

    def __init__(self, log: D4PyEventLog, declare_model: DeclareModel, consider_vacuity: bool, completed: bool):
        super().__init__(log, declare_model, consider_vacuity)
        self.completed = completed

    def run(self) -> MPDeclareResultsBrowser:
        if self.event_log is None:
            raise RuntimeError("You must load the log before checking the model.")
        if self.process_model is None:
            raise RuntimeError("You must load the DECLARE model before checking the model.")

        log_checkers_results = []
        for trace in self.event_log.get_log():
            log_checkers_results.append(Constraint_checker().check_trace_conformance(trace, self.process_model, self.completed,
                                                                                    self.consider_vacuity,
                                                                                    self.event_log.activity_key))
        return MPDeclareResultsBrowser(log_checkers_results, self.process_model.serialized_constraints)

class Constraint_checker ():
    def check_trace_conformance(self, trace: dict, decl_model: DeclareModel, completed: bool= True, consider_vacuity: bool = False,
                                concept_name: str = "concept:name") -> List[CheckerResult]:

        # Set containing all constraints that raised SyntaxError in checker functions
        rules = {"vacuous_satisfaction": consider_vacuity}
        error_constraint_set = set()
        model: DeclareModel = decl_model
        trace_results = []
        for idx, constraint in enumerate(model.constraints):
            constraint_str = model.serialized_constraints[idx]
            rules["activation"] = constraint['condition'][0]
            if constraint['template'].supports_cardinality:
                rules["n"] = constraint['n']
            if constraint['template'].is_binary:
                rules["correlation"] = constraint['condition'][1]
            rules["time"] = constraint['condition'][-1]  # time condition is always at last position
            try:
                trace_results.append(TemplateConstraintChecker(trace, completed, constraint['activities'], rules,
                                                               concept_name).get_template(constraint['template'])())
            except SyntaxError:
                # TODO: use python logger
                if constraint_str not in error_constraint_set:
                    error_constraint_set.add(constraint_str)
                    print('Condition not properly formatted for constraint "' + constraint_str + '".')
        return trace_results

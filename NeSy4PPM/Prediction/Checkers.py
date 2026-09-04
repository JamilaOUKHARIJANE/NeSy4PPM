from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.Utils.Declare.TraceStates import TraceState
from datetime import timedelta
from typing import List, Optional

from Declare4Py.Utils.Declare.Checkers import CheckerResult
from Declare4Py.Utils.Declare.Checkers import TemplateConstraintChecker as Declare4PyTemplateConstraintChecker


glob = {'__builtins__': None}


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
                if constraint_str not in error_constraint_set:
                    error_constraint_set.add(constraint_str)
                    print('Condition not properly formatted for constraint "' + constraint_str + '".')
        return trace_results

class TemplateConstraintChecker(Declare4PyTemplateConstraintChecker):
    def get_template(self, template):
        template_checker_name = f"mp{template.templ_str.replace(' ', '').replace('-', '')}"
        try:
            return getattr(self, template_checker_name)
        except AttributeError:
            print(f"The checker function for template {template.templ_str} has not been implemented yet.")

    @staticmethod
    def _sum_result_values(result_a: CheckerResult, result_b: CheckerResult, attr_name: str) -> Optional[int]:
        value_a = getattr(result_a, attr_name)
        value_b = getattr(result_b, attr_name)
        if value_a is None and value_b is None:
            return None
        return (value_a or 0) + (value_b or 0)

    def _combine_conjunction_results(self, result_a: CheckerResult, result_b: CheckerResult) -> CheckerResult:
        if result_a.state == TraceState.VIOLATED or result_b.state == TraceState.VIOLATED:
            state = TraceState.VIOLATED
        elif not self.completed:
            if result_a.state == TraceState.POSSIBLY_VIOLATED or result_b.state == TraceState.POSSIBLY_VIOLATED:
                state = TraceState.POSSIBLY_VIOLATED
            else:
                state = TraceState.POSSIBLY_SATISFIED
        else:
            state = TraceState.SATISFIED

        return CheckerResult(
            num_fulfillments=self._sum_result_values(result_a, result_b, "num_fulfillments"),
            num_violations=self._sum_result_values(result_a, result_b, "num_violations"),
            num_pendings=self._sum_result_values(result_a, result_b, "num_pendings"),
            num_activations=self._sum_result_values(result_a, result_b, "num_activations"),
            state=state,
        )

    def _run_with_reversed_activities_and_conditions(self, method_name: str) -> CheckerResult:
        rules = self.rules.copy()
        rules["activation"], rules["correlation"] = rules["correlation"], rules["activation"]
        checker = TemplateConstraintChecker(self.traces, self.completed, list(reversed(self.activities)), rules,
                                            self.concept_name)
        return getattr(checker, method_name)()

    def mpSuccession(self) -> CheckerResult:
        return self._combine_conjunction_results(self.mpResponse(), self.mpPrecedence())

    def mpAlternateSuccession(self) -> CheckerResult:
        return self._combine_conjunction_results(self.mpAlternateResponse(), self.mpAlternatePrecedence())

    def mpChainSuccession(self) -> CheckerResult:
        return self._combine_conjunction_results(self.mpChainResponse(), self.mpChainPrecedence())

    def mpCoExistence(self) -> CheckerResult:
        return self._combine_conjunction_results(
            self.mpRespondedExistence(),
            self._run_with_reversed_activities_and_conditions("mpRespondedExistence"),
        )

    def mpNotSuccession(self) -> CheckerResult:
        return self.mpNotResponse()

    def mpNotCoExistence(self) -> CheckerResult:
        return self.mpNotRespondedExistence()

    def mpNotChainSuccession(self) -> CheckerResult:
        return self.mpNotChainResponse()

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.StochasticDFA.dfa import SymbolicDFA
from NeSy4PPM.StochasticDFA.declare_model import Model, clean_activity_name, clean_resource_name


EDGE_RE = re.compile(
    r'^(?P<prefix>\s*(?P<src>\w+)\s*->\s*(?P<dst>\w+)\s*\[label=")'
    r'(?P<label>(?:\\"|[^"])*)'
    r'(?P<suffix>"[^\]]*\];\s*)$'
)

NODE_RE = re.compile(
    r'^\s*(?P<state>\w+)\s*\[(?P<attrs>[^\]]*)\];\s*$'
)

NODE_STYLE_STATES_RE = re.compile(
    r'^\s*node\s*\[(?P<attrs>[^\]]*)\]\s*;\s*(?P<states>.*)$'
)

END_SYMBOL = "end"
DOT_KEYWORDS = {"graph", "node", "edge", "init"}


def unescape_dot_label(label: str) -> str:
    return label.replace(r'\"', '"').replace(r"\n", "\n")


def escape_dot_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', r'\"').replace("\n", r"\n")


def is_accepting_node_attrs(attrs: str) -> bool:
    normalized_attrs = re.sub(r"\s+", "", attrs)
    return "doublecircle" in normalized_attrs or "peripheries=2" in normalized_attrs


def sort_state_key(state: str) -> tuple[int, int | str]:
    return (0, int(state)) if state.isdigit() else (1, state)


def read_dfa(
    dot_path: Path,
) -> tuple[
    list[str],
    set[str],
    dict[tuple[str, str], str],
    set[str],
    str,
    bool,
]:
    """Read DFA edges, accepting states, initial state and resource usage."""
    lines = dot_path.read_text(encoding="utf-8").splitlines(keepends=True)
    transitions: dict[tuple[str, str], str] = {}
    states: set[str] = set()
    accepting_states: set[str] = set()
    initial_state = "0"
    uses_resources = False

    for line in lines:
        edge_match = EDGE_RE.match(line)
        if edge_match:
            src = edge_match.group("src")
            dst = edge_match.group("dst")
            label = unescape_dot_label(edge_match.group("label"))

            if src == "init":
                states.add(dst)
                initial_state = dst
                continue

            states.update((src, dst))
            transitions[(src, label)] = dst
            continue

        node_match = NODE_RE.match(line)
        if node_match:
            state = node_match.group("state")
            if state in DOT_KEYWORDS:
                continue
            states.add(state)
            attrs = node_match.group("attrs")
            if is_accepting_node_attrs(attrs):
                accepting_states.add(state)
            continue

        node_style_match = NODE_STYLE_STATES_RE.match(line)
        if node_style_match and is_accepting_node_attrs(
            node_style_match.group("attrs")
        ):
            styled_states = set(
                re.findall(
                    r"\b(?!node\b)(?!edge\b)(?!graph\b)\w+\b",
                    node_style_match.group("states"),
                )
            )
            states.update(styled_states)
            accepting_states.update(styled_states)

    return lines, states, transitions, accepting_states, initial_state


def read_log_tokens(log_path: Path) -> tuple[list[list[str]], dict[str, str]]:
    """Read XES traces without appending an explicit end symbol."""
    tree = ET.parse(log_path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"

    traces: list[list[str]] = []
    display_labels: dict[str, str] = {}

    for trace in root.findall(f"{namespace}trace"):
        tokens: list[str] = []

        for event in trace.findall(f"{namespace}event"):
            activity = None
            activity_display = None
            resource = None
            resource_display = None

            for string_attr in event.findall(f"{namespace}string"):
                key = string_attr.attrib.get("key")
                value = string_attr.attrib.get("value", "")
                if key == "concept:name":
                    activity = clean_activity_name(value)
                    activity_display = value

            if activity is None:
                continue
            token = activity
            display_labels.setdefault(token, activity_display or activity)

            tokens.append(token)

        traces.append(tokens)

    return traces, display_labels


def compute_runtime_dfa(
    transitions: dict[tuple[str, str], str],
    accepting_states: set[str],
    end_symbol: str = "end",
):

    """
    Build runtime DFA for SDFA.

    - Every non-end DFA transition is preserved as an SDFA transition.
    - End transitions into accepting states define termination-enabled states.
    """

    # states where termination is allowed
    termination_allowed_states = {
        src
        for (src, label), dst in transitions.items()
        if label == end_symbol
        and dst in accepting_states
    }


    if not termination_allowed_states:
        raise ValueError(
            "No termination states found."
        )

    ordinary_transitions = {
        (src, label): dst
        for (src, label), dst in transitions.items()
        if label != end_symbol
    }

    return (
        ordinary_transitions,
        termination_allowed_states
    )

def replay_traces(
    traces: list[list[str]],
    ordinary_transitions: dict[tuple[str, str], str],
    termination_allowed_states: set[str],
    initial_state: str,
) -> tuple[
    Counter[tuple[str, str, str]],
    Counter[str],
    Counter[tuple[str, str]],
    Counter[str],
]:
    """Replay traces and count ordinary transitions and valid terminations."""
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    termination_counts: Counter[str] = Counter()
    missing: Counter[tuple[str, str]] = Counter()
    invalid_termination: Counter[str] = Counter()

    for trace in traces:
        state = initial_state
        valid = True

        for token in trace:
            dst = ordinary_transitions.get((state, token))
            if dst is None:
                missing[(state, token)] += 1
                valid = False
                break

            edge_counts[(state, dst, token)] += 1
            state = dst

        if valid:
            if state in termination_allowed_states:
                termination_counts[state] += 1
            else:
                invalid_termination[state] += 1

    return edge_counts, termination_counts, missing, invalid_termination


def compute_probabilities(
    states: set[str],
    ordinary_transitions: dict[tuple[str, str], str],
    edge_counts: Counter[tuple[str, str, str]],
    termination_counts: Counter[str],
    termination_allowed_states: set[str],
) -> tuple[dict[tuple[str, str, str], float], dict[str, float]]:
    """
    Estimate transition probabilities jointly with implicit termination.

    For each state:
        delta_p(q,a) = c(q,a) / (sum_b c(q,b) + c_T(q))

    and tau(q) is serialized as:
        tau(q) = 1 - sum_a P(a|q).

    For states where termination is forbidden by the DFA, c_T(q)=0 and tau(q)=0.
    """
    outgoing: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for (src, label), dst in ordinary_transitions.items():
        outgoing[src].append((src, dst, label))

    transition_probabilities: dict[tuple[str, str, str], float] = {}
    implicit_tau: dict[str, float] = {}

    for state in states:
        edges = outgoing.get(state, [])
        activity_total = sum(edge_counts[edge] for edge in edges)
        termination_allowed = state in termination_allowed_states
        termination_total = termination_counts[state] if termination_allowed else 0

        denominator = activity_total + termination_total
        if denominator <= 0:
            if termination_allowed:
                for edge in edges:
                    transition_probabilities[edge] = 0.0
                implicit_tau[state] = 1.0
                continue
            for edge in edges:
                transition_probabilities[edge] = 0.0
            implicit_tau[state] = 0.0
            continue

        for edge in edges:
            transition_probabilities[edge] = edge_counts[edge] / denominator

        outgoing_sum = sum(transition_probabilities[edge] for edge in edges)
        tau = 1.0 - outgoing_sum if termination_allowed else 0.0

        if tau < -1e-12:
            raise ValueError(
                f"Negative implicit termination probability at state {state}: {tau}"
            )

        implicit_tau[state] = max(0.0, tau)

        total = outgoing_sum + implicit_tau[state]
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Probabilities at state {state} sum to {total}, not 1."
            )

    return transition_probabilities, implicit_tau

def write_sdfa(
    output_path: Path,
    states: set[str],
    accepting_states: set[str],
    ordinary_transitions: dict[tuple[str, str], str],
    transition_probabilities: dict[tuple[str, str, str], float],
    implicit_tau: dict[str, float],
    edge_counts: Counter[tuple[str, str, str]],
    activities_mapping,
    initial_state: str) -> None:
    """
    Write an SDFA that preserves the DFA state/transition structure.

    Termination probabilities are serialized explicitly so unobserved
    non-terminal states do not accidentally get tau(q)=1.
    """
    transitions_json: list[dict[str, object]] = []

    ordered_edges = sorted(
        (
            (src, dst, label)
            for (src, label), dst in ordinary_transitions.items()
        ),
        key=lambda edge: (int(edge[0]), int(edge[1]), edge[2]),
    )

    for src, dst, label in ordered_edges:
        edge = (src, dst, label)
        probability = transition_probabilities[edge]
        transitions_json.append(
            {
                "from": int(src),
                "to": int(dst),
                "label": activities_mapping[label], #display_labels.get(label, label),
                "prob": f"{probability:.12f}",
                "observedCount": int(edge_counts[edge]),
            }
        )

    transition_lines = [
        "    " + json.dumps(transition, ensure_ascii=False)
        for transition in transitions_json
    ]
    if transition_lines:
        transition_lines = [
            line + ("," if index < len(transition_lines) - 1 else "")
            for index, line in enumerate(transition_lines)
        ]

    termination_lines = [
        f'    "{int(state)}": "{implicit_tau.get(state, 0.0):.12f}"'
        for state in sorted(states, key=int)
    ]
    if termination_lines:
        termination_lines = [
            line + ("," if index < len(termination_lines) - 1 else "")
            for index, line in enumerate(termination_lines)
        ]

    output_lines = [
        "{",
        f'  "initialState": {int(initial_state)},',
        '  "states": ['
        + ", ".join(str(int(state)) for state in sorted(states, key=int))
        + "],",
        '  "acceptingStates": ['
        + ", ".join(str(int(state)) for state in sorted(accepting_states, key=int))
        + "],",
        '  "terminationProbabilities": {',
        *termination_lines,
        "  },",
        '  "transitions": [',
        *transition_lines,
        "  ]",
        "}",
    ]

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main(datasets = ["BPI2013_In","BPIC2012"],
         log_path = Path.cwd().parent.parent / "docs" / "source" / 'data' / 'Procedural' / 'input'/ "logs",
         declare_path=Path.cwd().parent.parent / "docs" / "source" / 'data' / 'Procedural' / 'input'/ 'declare_models') -> int:
    for dataset in datasets:
        log_data = LogData(log_path=log_path, log_name=dataset, train_log=dataset + f"_train.xes",
                           feedback_log=dataset + f"_feedback.xes",
                           test_log=dataset + "_test.xes")
        dfa_folder = os.path.join(log_path.parent,'DFA', dataset)
        labels = [clean_activity_name(activity) for activity in list(log_data.act_enc_mapping.values())]
        activities_mapping = {clean_activity_name(activity): activity for activity in list(log_data.act_enc_mapping.values())}
        symbolic_dfa = SymbolicDFA(labels, dfa_folder)
        if not os.path.exists(dfa_folder):
            os.makedirs(dfa_folder)
            declare_model = Model(declare_path, dataset)
            symbolic_dfa.build_from_formula(declare_model.to_ltl())
        else:
            symbolic_dfa.build_from_file()

        dot_path = (log_path.parent/'DFA'/ dataset / "simpleDFA_final.dot")
        feedback_log_path = log_path / f"{dataset}_feedback.xes"
        sdfa_output_path = (log_path.parent/'DFA'/ dataset / f"{dataset}.sdfa")

        ( lines, states,complete_transitions, accepting_states, initial_state) = read_dfa(dot_path)
        print("Initial state:", initial_state)
        print("Accepting states:", accepting_states)
        print("Number states:", len(states))
        print("Number transitions:", len(complete_transitions))

        ordinary_transitions,termination_allowed_states = compute_runtime_dfa(complete_transitions, accepting_states)
        traces, display_labels = read_log_tokens(feedback_log_path)
        edge_counts,termination_counts, missing, invalid_termination,= replay_traces(traces, ordinary_transitions, termination_allowed_states, initial_state)
        transition_probabilities, implicit_tau = compute_probabilities(states,ordinary_transitions,edge_counts,termination_counts,termination_allowed_states)
        write_sdfa(sdfa_output_path,states,accepting_states,ordinary_transitions,transition_probabilities,implicit_tau,edge_counts, activities_mapping, initial_state)

        print(f"Wrote {sdfa_output_path}")
        print(f"Replayed {len(traces)} traces")
        print(f"Preserved states: {sorted(states, key=int)}")
        print(f"Preserved activity transitions: {len(ordinary_transitions)}")
        print( "Termination-enabled states: " f"{sorted(termination_allowed_states, key=int)}")
        print("Implicit termination probabilities: "+ ", ".join(f"tau({state})={implicit_tau.get(state, 0.0):.12f}"
                for state in sorted(states, key=int)))
        print(f"Counted {sum(edge_counts.values())} observed activity transitions")
        print(f"Counted {sum(termination_counts.values())} valid terminations")

        if missing:
            print(f"Stopped {sum(missing.values())} traces early because no "
                "DFA edge matched:")
            for (state, token), count in missing.most_common(10):
                print(f"  state={state} token={token} count={count}")
            if len(missing) > 10:
                print(f"  ... {len(missing) - 10} more")

        if invalid_termination:
            print(f"Found {sum(invalid_termination.values())} traces ending in "
                "states where valid termination is not permitted:"
            )
            for state, count in invalid_termination.most_common(10):
                print(f"  state={state} count={count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os.path
import networkx as nx
from ltlf2dfa.parser.ltlf import LTLfParser
import torch
from torch import nn


class SymbolicDFA:
    def __init__(self, labels, folder_path):
        self.labels = labels
        self.folder_path = folder_path
        self.graph = nx.MultiDiGraph()
        self.initial_state = None
        self.accepting_state = None
        self.state_types = {}

    def build_from_formula(self, formula):
        parser = LTLfParser()
        ast = parser(formula)
        dot = ast.to_dfa()

        with open(os.path.join(self.folder_path, 'symbolicDFA.dot'), 'w+') as file:
            file.write(dot)

        self.build_from_file()

    def build_from_file(self):
        with open(os.path.join(self.folder_path, 'symbolicDFA.dot'), 'r') as file:
            dot = file.read()

        #token_symbols = symbols(self.labels)
        #token_map = dict(zip(self.labels, token_symbols))

        temp_accepting_states = []
        for line in dot.splitlines():
            if 'doublecircle' in line:
                finals = line.strip().split(';')[1:-1]
                temp_accepting_states = [int(s.strip()) - 1 for s in finals]
            elif '->' in line:
                if 'init' in line:
                    parts = line.strip().split() #split(' ')
                    self.initial_state = int(parts[2][:-1]) - 1
                else:
                    parts = line.strip().split()
                    src, dst = int(parts[0]) - 1, int(parts[2]) - 1
                    label = line.strip().split('"')[1]

                    #guard = sympify(a=label, locals=token_map)
                    for token in valid_tokens_for_guard(label, self.labels): #(guard, self.labels):
                        self.graph.add_edge(src, dst, token)

        initial_states = list(self.graph.nodes)
        self.accepting_state = max(self.graph.nodes) + 1
        final_rejecting = max(self.graph.nodes) + 2
        self.graph.add_node(self.accepting_state)

        for state in initial_states:
            if state in temp_accepting_states:
                self.graph.add_edge(state, self.accepting_state, 'end')
            else:
                self.graph.add_edge(state, final_rejecting, 'end')

        all_rejecting = self.extract_rejecting_states()
        for state in self.graph.nodes:
            if state == self.accepting_state:
                self.state_types[state] = 1
            elif state in all_rejecting:
                self.state_types[state] = -1
            else:
                self.state_types[state] = 0

        for label in self.labels + ['end']:
            self.graph.add_edge(final_rejecting, final_rejecting, label)
            self.graph.add_edge(self.accepting_state, self.accepting_state, label)

        self.write_final_dot_to_file()

    def extract_rejecting_states(self):
        rev_graph = self.graph.reverse(copy=False)
        reachable = {self.accepting_state}
        reachable.update(nx.descendants(rev_graph, self.accepting_state))
        return self.graph.nodes - reachable

    def to_deep_dfa(self, device):
        deep_dfa = DeepDFA(len(self.graph.nodes), len(self.labels) + 1, device)
        deep_dfa.build(self.state_types, self.graph.edges, self.labels)
        return deep_dfa

    def write_final_dot_to_file(self):
        intro = """digraph MONA_DFA {
rankdir = LR;
center = true;
size = "7.5,10.5";
edge [fontname = Courier];
node [height = .5, width = .5];
"""
        end = f'node [shape = doublecircle]; {self.accepting_state};'
        start = f'node [shape = circle]; {self.initial_state};\ninit [shape = plaintext, label = ""];\ninit -> {self.initial_state};'
        transitions_string = ""
        for src, dst, label in self.graph.edges:
            transitions_string += f'{src} -> {dst} [label="{label}"];\n'
        transitions_string += "}"

        with open(os.path.join(self.folder_path, 'simpleDFA_final.dot'), 'w+') as file:
            file.write(intro + end + '\n' + start + '\n' + transitions_string)


def valid_tokens_for_guard(guard_expr, tokens):
    if isinstance(guard_expr, str):
        valid_mask = _evaluate_guard_mask(guard_expr, tokens)
        return [token for i, token in enumerate(tokens) if valid_mask & (1 << i)]

    valid = []
    for token in tokens:
        assignment = {t: False for t in tokens}
        assignment[token] = True
        if bool(guard_expr.subs(assignment)):
            valid.append(token)
    return valid


def _evaluate_guard_mask(guard, tokens):
    token_masks = {token: 1 << i for i, token in enumerate(tokens)}
    universe_mask = (1 << len(tokens)) - 1
    values = []
    operators = []

    def apply_operator():
        operator = operators.pop()
        if operator == '~':
            values.append(universe_mask ^ values.pop())
            return

        right = values.pop()
        left = values.pop()
        if operator == '&':
            values.append(left & right)
        elif operator == '|':
            values.append(left | right)

    i = 0
    while i < len(guard):
        char = guard[i]
        if char.isspace():
            i += 1
            continue

        if char == '(':
            operators.append(char)
            i += 1
            continue

        if char == ')':
            while operators and operators[-1] != '(':
                apply_operator()
            if not operators:
                raise ValueError(f'Unbalanced guard expression: {guard[:100]}')
            operators.pop()
            while operators and operators[-1] == '~':
                apply_operator()
            i += 1
            continue

        if char in ('~', '!'):
            operators.append('~')
            i += 1
            continue

        if char in ('&', '|'):
            operator = char
            i += 2 if i + 1 < len(guard) and guard[i + 1] == char else 1
            while (
                operators
                and operators[-1] != '('
                and _GUARD_OPERATOR_PRECEDENCE[operators[-1]] >= _GUARD_OPERATOR_PRECEDENCE[operator]
            ):
                apply_operator()
            operators.append(operator)
            continue

        start = i
        while i < len(guard) and (guard[i].isalnum() or guard[i] == '_'):
            i += 1
        if start == i:
            raise ValueError(f'Unexpected character {guard[i]!r} in guard expression: {guard[:100]}')

        identifier = guard[start:i]
        normalized_identifier = identifier.lower()
        if normalized_identifier in ('true', '1'):
            values.append(universe_mask)
        elif normalized_identifier in ('false', '0'):
            values.append(0)
        elif identifier in token_masks:
            values.append(token_masks[identifier])
        else:
            raise ValueError(f'Unknown token {identifier!r} in guard expression')

        while operators and operators[-1] == '~':
            apply_operator()

    while operators:
        if operators[-1] == '(':
            raise ValueError(f'Unbalanced guard expression: {guard[:100]}')
        apply_operator()

    if len(values) != 1:
        raise ValueError(f'Invalid guard expression: {guard[:100]}')
    return values[0]


_GUARD_OPERATOR_PRECEDENCE = {'|': 1, '&': 2, '~': 3}


class DeepDFA(nn.Module):
    def __init__(self, n_states, n_actions, device):
        super(DeepDFA, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.trans_prob = torch.zeros((n_actions, n_states, n_states), requires_grad=False, device=device)
        self.accepting_matrix = torch.zeros((n_states, 2), requires_grad=False, device=device)
        self.rejecting_matrix = torch.zeros((n_states, 2), requires_grad=False, device=device)

    def to_device(self, device):
        device = torch.device(device)
        if any(tensor.device != device for tensor in (self.trans_prob, self.accepting_matrix, self.rejecting_matrix)):
            self.trans_prob = self.trans_prob.to(device)
            self.accepting_matrix = self.accepting_matrix.to(device)
            self.rejecting_matrix = self.rejecting_matrix.to(device)
        self.device = device
        return self

    def build(self, state_types, edges, labels):
        labels_map = {label: i for i, label in enumerate(labels + ['end'])}

        with torch.no_grad():
            for (src, dst, label) in edges:
                self.trans_prob[labels_map[label], src, dst] = 1.0

            for s in state_types:
                self.accepting_matrix[s, int(state_types[s] == 1)] = 1.0
                self.rejecting_matrix[s, int(state_types[s] == -1)] = 1.0
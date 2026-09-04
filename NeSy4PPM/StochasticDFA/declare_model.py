import os
import re


RESOURCE_ATTRIBUTE = 'org:resource'


class Model:
    def __init__( self, folder_path, dataset, resource=False, resource_names=None):
        self.folder_path = folder_path
        self.name = f"BK_{dataset}"
        self.resource = resource
        self.content = self.read_from_file()
        self.formulas = []
        self.resource_names = resource_names

    def to_ltl(self):
        formulas = []
        for row in self.content.split('\n'):
            if row and not row.startswith('#'):
                match = re.match(r'(.+?)\[(.*?)\]\s*(?:\|(.*))?$', row)
                if match:
                    args = [arg.strip() for arg in match.group(2).split(',')]
                    conditions = [condition.strip() for condition in match.group(3).split('|')] if match.group(3) else []
                    while len(conditions) < 1: # < 3 if we consider correlation and target conditions
                        conditions.append('')
                    target = args[1] if len(args) > 1 else None
                    declare_constraint = MPDeclareConstraint(match.group(1), args[0], target, conditions[0],self.resource, self.resource_names)
                    if declare_constraint:
                        ltl_constraint = declare_constraint.to_ltl()
                        if ltl_constraint:
                            formulas.append(ltl_constraint)
        self.formulas = formulas
        return ' & '.join(self.formulas)

    def read_from_file(self):
        with open(os.path.join(self.folder_path, f'{self.name}.decl')) as f:
            return f.read()

    def write_formula_to_file(self):
        with open(os.path.join(self.folder_path,f'{self.name}_ltl.txt'), 'w') as f:
            f.write(' &\n'.join(self.formulas))




class MPDeclareConstraint:
    def __init__(self, template, activator, target=None, activation_condition='',
                 resource=False, resource_names=None):
        self.name = template
        self.activation = None
        self.target = None
        self.resources = resource_names
        if resource:
            if self.name.endswith('Precedence'):
                self.activation = self.all_activity_resources(clean_activity_name(activator))
                self.target = activity_resource_activation(target, activation_condition)
            else:
                self.activation = activity_resource_activation(activator, activation_condition)
                if target: self.target = self.all_activity_resources(clean_activity_name(target))
        else:
            self.activation = clean_activity_name(activator)
            if target:
                self.target = clean_activity_name(target)

    def to_ltl(self):
        if not self.activation:
            return None

        if self.name == 'Init':
            return f'({self.activation})'
        elif self.name == 'Existence':
            return f'(F({self.activation}))'
        elif self.name == 'Existence2':
            return f'(F({self.activation} & X(F({self.activation}))))'
        elif self.name == 'Existence3':
            return f'(F({self.activation} & X(F({self.activation} & X(F({self.activation}))))))'
        elif self.name == 'Absence':
            return f'(!(F({self.activation})))'
        elif self.name == 'Absence2':
            return f'(!(F({self.activation} & X(F({self.activation})))))'
        elif self.name == 'Absence3':
            return f'(!(F({self.activation} & X(F({self.activation} & X(F({self.activation})))))))'
        elif self.name == 'Exactly1':
            return f'(F({self.activation}) & !(F({self.activation} & X(F({self.activation})))))'

        if not self.target:
            return None

        if self.name == 'Choice':
            return f'(F({self.activation}) | F({self.target}))'
        elif self.name == "Exclusive Choice":
            return f'((F({self.activation}) & !(F({self.target}))) | (F({self.target}) & !(F({self.activation}))))'
        elif self.name == 'Responded Existence':
            return f'(F({self.activation}) -> F({self.target}))'
        elif self.name == 'Co-Existence':
            return f'((F({self.activation}) -> F({self.target})) & (F({self.target}) -> F({self.activation})))'
        elif self.name == 'Response':
            return f'(G({self.activation} -> F({self.target})))'
        elif self.name == 'Alternate Response':
            return f'(G({self.activation} -> X(!({self.activation}) U {self.target})))'
        elif self.name == 'Chain Response':
            return f'(G({self.activation} -> X({self.target})))'
        elif self.name == 'Precedence':
            return f'((!({self.target}) U {self.activation}) | G(!({self.target})))'
        elif self.name == 'Alternate Precedence':
            #return f'(!{self.target} U {self.activation}) & G({self.target} ->X((!{self.target} U {self.activation}) | G(!({self.target})))'
            return f'((((!{self.target} U {self.activation}) | G(!{self.target})) & G({self.target} ->((!(X({self.activation})) & !(X(!({self.activation})))) | X((!({self.target}) U {self.activation}) | G(!({self.target})))))) & !({self.target}))'
        elif self.name == 'Chain Precedence':
            return f'(G(X({self.target}) -> {self.activation}))'
        elif self.name == 'Succession':
            return f'(G({self.activation} -> F({self.target})) & (!({self.target}) U {self.activation}) | G (!{self.target}))'
        elif self.name == 'Alternate Succession':
            return f'(G({self.activation} -> X(! {self.activation} U {self.target})) & (!({self.target}) U {self.activation}) | G(!{self.target}))'
        elif self.name == 'Chain Succession':
            return f'((G({self.activation} -> X({self.target}))) & (G(X({self.target}) -> {self.activation})))'
        elif self.name == 'Not Co-Existence':
            return f'(!(F({self.activation}) & F({self.target})))'
        elif self.name == 'Not Succession':
            return f'(G({self.activation} -> !(F({self.target}))))'
        elif self.name == 'Not Chain Succession':
            return f'(G({self.activation} -> !(X({self.target}))))'  # & (G(X(!({self.target})) -> {self.activation})))'
        else:
            return None

    def all_activity_resources(self, activity):
        labels = [f"{activity}__{resource}" for resource in self.resources]
        if len(labels) == 1:
            return labels[0]
        return '(' + ' | '.join(labels) + ')'


def activity_resource_activation(activity, condition=""):
    activity_token = clean_activity_name(activity)
    condition = condition.replace('"', '').replace("'", '')
    attr = re.escape(RESOURCE_ATTRIBUTE)
    values = []

    in_pattern = rf'(?:A\.|T\.)?{attr}\s+in\s*\(([^)]+)\)'
    for match in re.finditer(in_pattern, condition, flags=re.IGNORECASE):
        values.extend(v.strip() for v in match.group(1).split(','))

    eq_pattern = rf'(?:A\.|T\.)?{attr}\s*(?:is|=|==)\s*([^&|)]+)'
    for match in re.finditer(eq_pattern, condition, flags=re.IGNORECASE):
        values.append(match.group(1).strip())

    resources = list(dict.fromkeys(clean_resource_name(value) for value in values if value))
    if not resources:
        return activity_token

    labels = [f"{activity_token}__{resource}" for resource in resources]
    if len(labels) == 1:
        return labels[0]
    return '(' + ' | '.join(labels) + ')'

def clean_activity_name(name):
    return f"a_{name.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace(')', '_')}"

def clean_resource_name(name):
    return f"r_{name.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace(')', '_')}"
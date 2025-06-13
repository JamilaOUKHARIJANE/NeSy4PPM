from Declare4Py.ProcessModels.DeclareModel import DeclareModelTemplate


#Declare4Py.ProcessModels.LTLModel.parse_from_string(self, content: str, new_line_ctrl: str = "\n") parses 'X' into weak next, using 'X[!]' for next instead
@staticmethod
def get_constraint_formula(template: DeclareModelTemplate, ac1: str, ac2: str, cardinality: int) -> str:
    if template is DeclareModelTemplate.ABSENCE:
        if cardinality == 1:
            return "(!(F(" + ac1 + ")))"
        elif cardinality == 2:
            return "(!(F((" + ac1 + " && X[!](F(" + ac1 + "))))))"
        elif cardinality == 3:
            return "(!(F((" + ac1 + " && X[!](F((" + ac1 + " && X[!](F(" + ac1 + ")))))))))"
        else:
            print("unsupported cardinality")
            return None
    elif template is DeclareModelTemplate.ALTERNATE_PRECEDENCE:
        return "((((!(" + ac2 + ") U(" + ac1 + ")) || G(!(" + ac2 + "))) && G((" + ac2 + " -> ((!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + ")))) || X[!]((!(" + ac2 + ") U(" + ac1 + ")) || G(!(" + ac2 + "))))))) && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + "))))))"
    elif template is DeclareModelTemplate.ALTERNATE_RESPONSE:
        return "(G(" + ac1 + " -> X[!](((!(" + ac1 + ")) U(" + ac2 + ")))))"
    elif template is DeclareModelTemplate.ALTERNATE_SUCCESSION:
        return "(G((" + ac1 + " -> X[!]((!(" + ac1 + ") U(" + ac2 + "))))) && ((((!(" + ac2 + ") U(" + ac1 + ")) || G(!(" + ac2 + "))) && G(" + ac2 + " ->((!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + ")))) || X[!](((!(" + ac2 + ") U(" + ac1 + ")) || G(!(" + ac2 + "))))))) && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + ")))))))"
    elif template is DeclareModelTemplate.CHAIN_PRECEDENCE: #TODO: Double-check this formula, the automata seems slightly incorrect for chain precedence
        return "(G(X[!](" + ac2 + ") -> " + ac1 + ") && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + "))))))"
    elif template is DeclareModelTemplate.CHAIN_RESPONSE:
        return "(G(" + ac1 + " -> X[!](" + ac2 + ")))"
    elif template is DeclareModelTemplate.CHAIN_SUCCESSION: #TODO: Double-check this formula, the automata seems slightly incorrect for chain succession
        return "((G(" + ac1 + " -> X[!](" + ac2 + "))) && (G(X[!](" + ac2 + ") -> " + ac1 + ") && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + ")))))))"
    elif template is DeclareModelTemplate.CHOICE:
        return "(F(" + ac1 + ") || F(" + ac2 + "))"
    elif template is DeclareModelTemplate.CO_EXISTENCE:
        return "((F(" + ac1 + ") -> F(" + ac2 + ")) && (F(" + ac2 + ") -> F(" + ac1 + ")))"
    elif template is DeclareModelTemplate.END:
        return "(F(" + ac1 + " && !X[!]((" + ac1 + ") U(!" + ac1 + "))))"
    elif template is DeclareModelTemplate.EXACTLY:
        if cardinality == 1:
            return "(F(" + ac1 + ") && !(F((" + ac1 + " && X[!](F(" + ac1 + "))))))"
        elif cardinality == 2:
            return "(F(" + ac1 + " && (" + ac1 + " -> (X[!](F(" + ac1 + "))))) && !(F(" + ac1 + " && (" + ac1 + " -> X[!](F(" + ac1 + " && (" + ac1 + " -> X[!](F(" + ac1 + ")))))))))"
        else:
            print("unsupported cardinality")
            return None
    elif template is DeclareModelTemplate.EXCLUSIVE_CHOICE:
        return "((F(" + ac1 + ") || F(" + ac2 + ")) && !((F(" + ac1 + ") && F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.EXISTENCE:
        if cardinality == 1:
            return "(F(" + ac1 + "))"
        elif cardinality == 2:
            return "(F(" + ac1 + " && X[!](F(" + ac1 + "))))"
        elif cardinality == 3:
            return "(F(" + ac1 + " && X[!](F(" + ac1 + " && X[!](F(" + ac1 + "))))))"
        else:
            print("unsupported cardinality")
            return None
    elif template is DeclareModelTemplate.INIT:
        return "(" + ac1 + ")"
    elif template is DeclareModelTemplate.NOT_CHAIN_PRECEDENCE: #Double-check that ac1 and ac2 are in correct order
        return "(G(" + ac1 + " -> !(X[!](" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_CHAIN_RESPONSE:
        return "(G(" + ac1 + " -> !(X[!](" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_CHAIN_SUCCESSION:
        return "(G(" + ac1 + " -> !(X[!](" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_CO_EXISTENCE:
        return "((F(" + ac1 + ")) -> (!(F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_PRECEDENCE:
        return "(G(" + ac1 + " -> !(F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_RESPONDED_EXISTENCE:
        return "((F(" + ac1 + ")) -> (!(F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_RESPONSE:
        return "(G(" + ac1 + " -> !(F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.NOT_SUCCESSION:
        return "(G(" + ac1 + " -> !(F(" + ac2 + "))))"
    elif template is DeclareModelTemplate.PRECEDENCE:
        return "((!(" + ac2 + ") U(" + ac1 + ")) || (G(!(" + ac2 + "))) && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + "))))))"
    elif template is DeclareModelTemplate.RESPONDED_EXISTENCE:
        return "(F(" + ac1 + ") -> (F(" + ac2 + ")))"
    elif template is DeclareModelTemplate.RESPONSE:
        return "(G(" + ac1 + " -> F(" + ac2 + ")))"
    elif template is DeclareModelTemplate.SUCCESSION:
        return "((G(" + ac1 + " -> F(" + ac2 + "))) && ((!(" + ac2 + ") U(" + ac1 + ")) || (G(!(" + ac2 + "))) && (!(" + ac2 + ") || (!(X[!](" + ac1 + ")) && !(X[!](!(" + ac1 + ")))))))"
    else:
        print("Unsupported template")
        return None




import random
import pdb

class Obj:
    def __init__(self, name, moreattrs={}, constraints={}):
        self.name = name
        self.constraints = constraints
        for attr in moreattrs:
            setattr(self, attr, moreattrs[attr])

pdb.set_trace()
RULES = {
    'INS': 'ACT OBJ[ACT]',
    'OBJ[ACT]': 'NN[ACT]|ATTR[NN]',
    'ATTR': 'COLOR,LOC,STATE',
    'LOC': 'LOC_ABS|LOC_REL'

CONCEPTS = {
    'NN': ['WALL', 'DOOR', 'KEY', 'BOX', 'BALL'],
    'ACT': ['GOTO', 'OPEN', 'PICK', 'DROP'],
    'COLOR': ['RED', 'GREEN', 'BLUE'],
    'LOC_ABS': ['EAST', 'WEST', 'SOUTH', 'NORTH'],
    'LOC_REL': ['LEFT', 'RIGHT'],
    'STATE': ['LOCKED']}

CONSTRAINTS = {
        'WALL': ['GOTO'],
        'DOOR': ['GOTO', 'OPEN', 'LOCKED'],
        'KEY': ['GOTO', 'PICK', 'DROP'],
        'BOX': ['GOTO', 'OPEN', 'PICK', 'DROP'],
        'BALL': ['GOTO', 'PICK', 'DROP']}

SURFACES = {
    # Any concepts that is not in this dic will directly convert to its lower-case.
    # ACTs
    'GOTO': ['go to', 'reach', 'find', 'walk to'],
    'PICK': ['pick', 'pick up', 'grasp', 'go pick', 'go grasp', 'get', 'fetch', 'find'],
    'DROP': ['drop', 'drop down'],
    # ATTRs
    'COLOR:NN': ['COLOR NN', 'NN with the color of COLOR', 'NN in COLOR', 'NN which is in COLOR', 'NN that is in COLOR'],
    'LOC_ABS:NN': ['LOC_ABS NN', 'NN on the LOC_ABS', 'NN on the LOC_ABS direction', 'NN which is on the LOC_ABS'],
    'LOC_REL:NN': ['LOC_REL NN', 'NN on the LOC_REL', 'NN on your LOC_REL'],
    'STATE:NN': ['STATE NN', 'NN that is STATE', 'NN which is STATE'],
    'NN': ['the NN']}


def handle_brackets(instr):
    bracket_stack = []
    count_bracket = 1
    dict_bracket = {}
    new_bracket = True 
    pos_start = 0
    pos_end = 0
    instr_new = ''
    for i in range(len(instr)):
        if instr[i] != '(' and instr[i] != ')' and new_bracket:
            instr_new += instr[i]
        if instr[i] == '(':
            bracket_stack.append('(')
            if new_bracket:
                pos_start = i
                new_bracket = False
        if instr[i] == ')':
            if len(bracket_stack) == 0:
                raise ValueError
            else:
                bracket_stack.pop()
                if len(bracket_stack) == 0:
                    pos_end = i
                    dict_bracket['_{}_'.format(count_bracket)] = instr[pos_start+1:pos_end]
                    instr_new += '_{}_'.format(count_bracket)
                    new_bracket = True
                    count_bracket += 1
    return instr_new, dict_bracket

def parse(instr, keywords_full):

    instr_new, dict_bracket = handle_brackets(instr)

    value_space = instr_new.split(' ')
    value_space_ = []
    for v_sp in value_space:
        v_comma = v_sp.split(',')
        v_comma_must = []
        v_comma_rest = []
        for v_co in v_comma:
            if has_key_in(v_co, keywords_full):
                v_comma_must.append(v_co)
            else: 
                v_comma_rest.append(v_co)
        v_comma_rest = random.sample(v_comma_rest, random.randint(0, len(v_comma_rest)))
        v_comma = v_comma_must + v_comma_rest

        pdb.set_trace()
        v_comma_ = []
        for v_co in v_comma: 
            v_bar = v_co.split('|')
            v_bar_random = True
            for v_b in v_bar:
                if v_b in keywords_full:
                    v_bar = v_b
                    v_bar_random = False
                    break
            if v_bar_random:
                v_bar = random.choice(v_comma)

            if v_bar in dict_bracket:
                v_bar = parse(dict_bracket[v_bar], keywords_full)
            if v_bar in RULES:
                v_bar = parse(RULES[v_bar], keywords_full)

            v_comma_.append(v_bar)
        v_comma_ = ','.join(v_comma_)
        value_space_.append(v_comma_)
    print(' '.join(value_space_))
    return ' '.join(value_space_)

def has_key_in(x, keys):
    for k in keys:
        if k in x:
            return True
    return False

def instr_keywords_full(keywords):
    return_now = True 
    for w in keywords:
        for r_k in RULES:
            if w in RULES[r_k] and r_k not in keywords:
                keywords.append(r_k)
                return_now = False
        for c_k in CONCEPTS:
            if w in CONCEPTS[c_k] and c_k not in keywords:
                keywords.append(c_k)
                return_now = False
    return keywords if return_now else instr_keywords_full(keywords) 
    

def generate_instruction(instr_keywords_seq=['PICK RED LOC_ABS']):
    for instr_keywords in instr_keywords_seq:
        kw_full = instr_keywords_full(instr_keywords.split(' '))
        instr = 'INS'
        instr = parse(instr, kw_full)
   

if __name__ is not "__main__":
    kw_full = instr_keywords_full('PICK RED LOC_ABS'.split(' '))
    print(parse('INS', kw_full))
    pdb.set_trace()
    
Instr = namedtuple("Instr", ["action", "object"])
Object = namedtuple("Object", ["type", "color", "loc", "state"])

Ins(act=PAct, obj=PObj)
Ins(act="pick", obj=Obj(attr=PAttr, nn=PNn))
Ins(Act("pick"), Obj(Attr, "key"))
Ins(Act("pick"), Obj(Attr(Color("red"), Loc(None), State(None)), "key"))

CONSTRAINTS = { \
    'key': {'goto', 'pick', 'drop'},
    'wall': {'goto'},
    'door': {'goto', 'open', 'locked', 'state'},
    'ball': {'goto', 'pick', 'drop'},
    'box': {'goto', 'pick', 'drop', 'open'}}


def extract_constraint_objects(name):
    check_valid_concept(name)
    if root_concepts(name) == {'object'}:
        return {name}
    if root_concepts(name) in [{'action'}, {'attr'}]:
        constraint_objects = set()
        for key in CONSTRAINTS:
            if name in CONSTRAINTS[key]:
                constraint_objects |= {key}
        for pa in parent_concepts(name):
            constraint_objects |= extract_constraint_objects(pa)
        return constraint_objects
    raise ValueError("{}: not availble for extracting constraint objects.".format(name))

def extract_constraint_objects_byset(nameset):
    if not nameset:
        return CONCEPT['object']
    else:
        constraint_objects = set()
        for name in nameset:
            constraint_objects |= extract_constraint_objects(name)
        return constraint_objects

def extract_full_constraints(constraint_list):
    action_cands = set()
    attr_cands = set()
    object_cands = set()
    for cons in constraint_list:
        check_valid_concept(cons)
        action_cands |= {cons} if root_concepts(cons) == {'action'} else set()
        attr_cands |= {cons} if root_concepts(cons) == {'attr'} else set()
        object_cands |= {cons} if root_concepts(cons) == {'object'} else set()
    object_cands = extract_constraint_objects_byset(action_cands) & extract_constraint_objects_byset(attr_cands) & extract_constraint_objects_byset(object_cands) 
    if not object_cands or len(object_cands) > 1 or len(object_cands) > 1:
        raise ValueError('{}: constraints provided are unavailable to generate the instruction. Please check your logic.'.format(constraint_list))
    
    object_cands = extract_constraint_objects_byset(action_cands|attr_cands|object_cands)


def extract_constraint_dict(constraints):
    constraints_dict = defaultdict(lambda: [])
    for c in constraints:
        c_type = get_concept_type(c)
        constraints_dict[c_type] += [c]
    return constraints_dict

def get_constraint(name):
    if name in CONSTRAINTS:
        return CONSTRAINTS[name]
    constraint = []
    for k in CONSTRAINTS:
        if name in CONSTRAINTS[k]:
           constraint.append(k) 
    return constraints 

def get_full_constraints(constraints, remain_concept_types=list(deepcopy(CONCEPTS.keys()))):
    if constraints == []:
        return constraints

    constraint_dict = extract_constraint_dict(constraints)
    for c_type in constraints_dict:
        remain_concept_types.remove(c_type)

    constraints_new = []
    for c in constraints:
        c_cons_dict = extract_constraint_dict(get_constraint(c))
        for c_c_type in c_cons_dict:
            if c_c_type in remain_concept_types:
                c_cons.append(c_c)
                remain_concept_types.remove(c_c_type)
        constraints_new += c_cons
    return constraints + constraints_new



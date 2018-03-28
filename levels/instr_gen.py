__author__ = "Saizheng Zhang"
__credits__ = "Saizheng Zhang, Lucas Willems, Thien Huu Nguyen"

from collections import namedtuple, defaultdict
from copy import deepcopy
import itertools
import random
from .instrs import *

CONCEPTS = {
    'action': {'pick', 'goto', 'drop', 'open'},
    'object': {'door', 'wall', 'ball', 'key', 'box'},
    'attr': {'color', 'loc', 'state'},
    'loc': {'loc_abs', 'loc_rel'},
    'color': {'red', 'blue', 'green'},
    'loc_abs': {'east', 'west', 'north', 'south'},
    'loc_rel': {'left', 'right'},
    'state': {'locked'}}

# constraints (a, b) means that a and b can appear at the same time.
CONSTRAINTS = \
    {('key', v) for v in {'goto', 'pick', 'drop'}} | \
    {('wall', 'goto')} | \
    {('door', v) for v in {'goto', 'open', 'locked'}} | \
    {('ball', v) for v in {'goto', 'pick', 'drop'}} | \
    {('box', v) for v in  {'goto', 'pick', 'drop', 'open', 'locked'}} | \
    {('object', 'color'), ('object', 'loc')}

def check_valid_concept(name):
    if name in CONCEPTS:
        return True
    for c in CONCEPTS:
        if name in CONCEPTS[c]:
            return True
    raise ValueError("Incorrect concept name: {}".format(name))

def parent_concepts(name):
    check_valid_concept(name)
    parents = set()
    for k in CONCEPTS:
        if name in CONCEPTS[k]:
            parents = parents|{k}
    return parents

def ancestor_concepts(name):
    parent_c = parent_concepts(name)
    if parent_c == set():
        return set()
    else:
        ancestor_c = set()
        for pa in parent_c:
            ancestor_c |= ancestor_concepts(pa)
        return parent_concepts(name)| ancestor_c

def is_ancestor(x, y):
    return True if x in ancestor_concepts(y) else False

def child_concepts(name):
    check_valid_concept(name)
    return CONCEPTS[name]

def root_concepts(name):
    check_valid_concept(name)
    if parent_concepts(name) == set():
        return {name}
    else:
        roots = set()
        for pa in parent_concepts(name):
            roots |= root_concepts(pa)
        return roots

def is_consistent(m, n):
    check_valid_concept(m)
    check_valid_concept(n)

    # ancestor reduction rule
    def rule_anc_reduct(x, y):
        prod_xy = itertools.product(ancestor_concepts(x)|{x}, ancestor_concepts(y)|{y})
        if any([(p_xy in CONSTRAINTS) or ((p_xy[1], p_xy[0]) in CONSTRAINTS) for p_xy in prod_xy]):
            return True
        else:
            return False

    if rule_anc_reduct(m, n):
        return True

    # action-object-attribute rule
    def rule_act_obj_attr(x, y):
        for o in CONCEPTS['object']:
            if rule_anc_reduct(x, o) and rule_anc_reduct(y, o) and \
                root_concepts(x) in [{'action'}, {'attr'}] and \
                root_concepts(y) in [{'action'}, {'attr'}] and \
                root_concepts(x) != root_concepts(y):
                return True
        return False

    if rule_act_obj_attr(m, n):
        return True

    # exclusive rule
    def rule_exclusive(x, y):
        if x == y:
            return True
        if is_ancestor(x, y) or is_ancestor(y, x):
            return True

        exclusive_list = {'action', 'color', 'loc'}
        if ancestor_concepts(x) & ancestor_concepts(y) & exclusive_list:
            return False
        if 'attr' in ancestor_concepts(x) and 'attr' in ancestor_concepts(y):
            return True
        return False

    if rule_exclusive(m, n):
        return True

    return False

def extract_cands_in_generate(type, constraints=set()):
    cands = []
    for t in CONCEPTS[type]:
        if all([is_consistent(t, c) for c in constraints]) or not constraints:
            cands.append(t)
    return cands

def gen_instr_seq(seed, constraintss=[set()]):
    random.seed(seed)
    return [gen_ainstr(constraints) for constraints in constraintss]

def gen_ainstr(constraints=set()):
    act = gen_action(constraints)
    obj = gen_object(act, constraints)
    return Instr(action=act, object=obj)

def gen_action(constraints=set()):
    action_cands = extract_cands_in_generate('action', constraints)
    action = random.choice(action_cands)
    return action

def gen_object(act=None, constraints=set()):
    o_cands = extract_cands_in_generate('object', constraints|(set() if not act else {act}))
    o = random.choice(o_cands)

    o_color = gen_color(constraints=constraints)
    o_loc = gen_loc(constraints=constraints)
    o_state = gen_state(obj=o, constraints=constraints)

    return Object(type=o, color=o_color, loc=o_loc, state=o_state)

def gen_subattr(type, constraints=set()):
    cands = extract_cands_in_generate(type, constraints)
    if not cands:
        return None

    if any([is_ancestor(type, c) or type == c for c in constraints]):
        return random.choice(cands)
    else:
        if random.choice([True, False]):
            return random.choice(cands)
        else:
            return None

def gen_color(obj=None, constraints=set()):
    return gen_subattr('color', constraints)

def gen_loc(obj=None, act=None, constraints=set()):
    subloc = gen_subattr('loc', constraints)
    if not subloc:
        return None
    if subloc == 'loc_abs':
        return gen_locabs(obj=obj, act=act, constraints=constraints|{'loc_abs'})
    if subloc == 'loc_rel':
        return gen_locrel(obj=obj, act=act, constraints=constraints|{'loc_rel'})

def gen_locabs(obj=None, act=None, constraints=set()):
    return gen_subattr('loc_abs', constraints)

def gen_locrel(obj=None, act=None, constraints=set()):
    return gen_subattr('loc_rel', constraints)

def gen_state(obj=None, act=None, constraints=set()):
    return gen_subattr('state', constraints|(set() if not obj else {obj}))

def gen_surface(ntup, seed=0, conditions={}):
    if ntup == None:
        return ''

    if isinstance(ntup, list):
        random.seed(seed)
        s_instr = ''
        for i, ainstr in enumerate(ntup):
            if i > 0:
                s_instr += random.choice([' and then', ', then']) + ' '
            s_instr += gen_surface(ainstr)
        return s_instr

    if isinstance(ntup, Instr):
        s_ainstr = gen_surface(ntup.action)
        if ntup.action != 'drop':
            s_ainstr += ' ' + gen_surface(ntup.object)
        return s_ainstr

    if isinstance(ntup, Object):
        s_obj = ntup.type
        s_attrs = list([ntup.color, ntup.loc, ntup.state])
        random.shuffle(s_attrs)
        for f in s_attrs:
            if not f:
                continue
            cond = random.choice(['pre', 'after', 'which is', 'that is'])
            if cond == 'pre':
                s_obj = gen_surface(f, conditions={cond}) + ' ' + s_obj
            if cond == 'after':
                if 'which is' in s_obj or 'that is' in s_obj:
                    s_obj = s_obj + ' and is ' + gen_surface(f, conditions={cond})
                else:
                    s_obj = s_obj + ' ' + (('which is ' + gen_surface(f, conditions={cond})) if f in CONCEPTS['state'] else gen_surface(f, conditions={cond}))
            if cond in ['which is', 'that is']:
                if 'which is' in s_obj or 'that is' in s_obj:
                    s_obj = s_obj + ' and is ' + gen_surface(f, conditions={cond})
                else:
                    s_obj = s_obj + ' {} '.format(cond) + gen_surface(f, conditions={cond})
        return 'the '+ s_obj

    if ntup == 'goto':
        return random.choice(['go to', 'reach', 'find', 'walk to'])

    if ntup == 'pick':
        return random.choice(['pick', 'pick up', 'grasp', 'go pick', 'go grasp', 'go get', 'get', 'go fetch', 'fetch'])

    if ntup == 'drop':
        return random.choice(['drop', 'drop down', 'put down'])

    if ntup == 'open':
        return random.choice(['open'])

    if ntup in CONCEPTS['color']:
        if {'pre'} & conditions:
            return ntup
        if {'after'} & conditions:
            return random.choice(['with the color of {}'.format(ntup), 'in {}'.format(ntup)])
        if {'which is', 'that is'} & conditions:
            return 'in {}'.format(ntup)

    if ntup in CONCEPTS['loc_abs']:
        if {'pre'} & conditions:
            return ntup
        if {'after', 'which is', 'that is'} & conditions:
            return random.choice(['on the {}'.format(ntup), 'on the {} direction'.format(ntup)])

    if ntup in CONCEPTS['loc_rel']:
        if {'pre'} & conditions:
            return ntup
        if {'which is', 'that is', 'after'} & conditions:
            return random.choice(['on the {}'.format(ntup), 'on your {}'.format(ntup)])

    if ntup in CONCEPTS['state']:
        return random.choice([ntup])

def test():
    for i in range(10):
        seed = i
        instr = gen_instr_seq(seed)
        #print(instr)
        gen_surface(instr, seed)

    for i in range(10):
        instr = gen_instr_seq(i, constraintss=[{"pick", "key"}, {"drop"}])
        #print(instr)
        gen_surface(instr, seed)

    # Same seed must yield the same instructions and string
    str1 = gen_surface(gen_instr_seq(seed), seed=7)
    str2 = gen_surface(gen_instr_seq(seed), seed=7)
    assert str1 == str2

if __name__ == "__main__":
    test()

import copy
import argparse
import itertools
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#########

def generate_all_unique_sublists(args, n_tasks):
    """Generate a set of random task vectors"""
    all_combinations = list(itertools.product(range(0, args.base), repeat=args.n_var))
    if len(all_combinations) < n_tasks:
        raise ValueError("Not enough unique combinations available.")
    selected_combinations = random.sample(all_combinations, n_tasks)
    return [list(combination) for combination in selected_combinations]


def generate_all_unique_sublists_givenWs(exist_combinations, args, n_tasks):
    """Generate a new set of random task vectors"""
    all_combinations = list(itertools.product(range(0, args.base), repeat=args.n_var))
    exist_combinations = [tuple(combination) for combination in exist_combinations]
    all_combinations = list(set(all_combinations) - set(exist_combinations))
    if len(all_combinations) < n_tasks:
        raise ValueError("Not enough unique combinations available.")
    selected_combinations = random.sample(all_combinations, n_tasks)
    return [list(combination) for combination in selected_combinations]


def parallelogram_tasks_with_shared_components(Ws_unique, args):
    """Currently only work for 2 variables"""
    def generate_lists(a, b, p):
        while True:
            x = random.choice([i for i in range(p) if i != a and i != b])
            y = random.choice([i for i in range(p) if i != a and i != b and i != x])

            list1 = [a, y]
            list2 = [x, b]
            list3 = [x, y]

            if list1 not in Ws and list2 not in Ws and list3 not in Ws:
                return [list1, list2, list3]

    Ws = copy.deepcopy(Ws_unique)
    while len(Ws) < 4 * len(Ws_unique):
        for W in Ws_unique:
            if len(Ws) >= 4 * len(Ws_unique):
                break
            new_Ws = generate_lists(W[0], W[1], args.p)
            Ws.extend(new_Ws)
    
    return Ws


def get_ood_lists(Ws, args):
    all_combinations = list(itertools.product(range(0, args.base), repeat=args.n_var))
    Ws = set(tuple(W) for W in Ws)
    ood_Ws = list(set(all_combinations) - Ws)
    random.shuffle(ood_Ws)
    return ood_Ws[:args.max_ood_tasks]
import numpy as np
from colorama import Fore, Style, init

# --------------------------------------------
# Inicializacija barv
# --------------------------------------------
init(autoreset=False)
colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
rs = Style.RESET_ALL  # reset barv

# ----------------------------
# Reduced Error Pruning (REP)
# ----------------------------

def _edge_for(node, attr_value):
    """Vrne oznako veje glede na vrednost atributa v vozlišču."""
    if node.split_value is not None:
        if attr_value is None or (isinstance(attr_value, float) and np.isnan(attr_value)):
            return None
        return "<=" if attr_value <= node.split_value else ">"
    else:
        if attr_value is None or (isinstance(attr_value, float) and np.isnan(attr_value)):
            return None
        return str(attr_value)


def _assign_node_ids(node, start_id=1):
    """Rekurzivno dodeli zaporedne ID-je vsem vozliščem drevesa."""
    node.id = start_id
    next_id = start_id + 1
    if node.children:
        for child in node.children.values():
            next_id = _assign_node_ids(child, next_id)
    return next_id


def _init_pruning_counters(node, all_classes):
    """Rekurzivno inicializira števec 'pruning_freq' v vseh vozliščih za vse razrede."""
    node.pruning_freq = {cls: 0 for cls in all_classes}
    node.error = 0
    if node.children:
        for child in node.children.values():
            _init_pruning_counters(child, all_classes)


def rep_node_update(node, sample, sample_class):
    """Posodobi števec v vozlišču (pruning_freq) za dan primer iz rezalne množice."""
    node.pruning_freq[sample_class] = node.pruning_freq.get(sample_class, 0) + 1

    if node.children:
        attr_value = sample.get(node.attr)
        edge = _edge_for(node, attr_value)
        if edge is not None:
            child = node.children.get(edge)
            if child:
                rep_node_update(child, sample, sample_class)


def rep_tree_process_pruning_set(tree_root, pruning_set, target):
    """Posodobi števce v vozliščih za vse primere v rezalni množici."""
    for _, row in pruning_set.iterrows():
        sample_class = row[target]
        rep_node_update(tree_root, row, sample_class)
    return tree_root


def rep_tree_prune(node, target, verbose_level=0):
    """Rekurzivno reže drevo po Reduced Error Pruning pravilu."""
    majority_class = node.majority_class
    errors_if_pruned = sum(v for c, v in node.pruning_freq.items() if c != majority_class)

    # Če je list
    if node.children is None:
        node.error = errors_if_pruned
        if verbose_level > 0:
            print(f"{node.color}List (ID={node.id}): e={node.error}{rs}")
        return node

    # Rekurzivno poreži poddrevesa
    total_subtree_error = 0
    for child in node.children.values():
        rep_tree_prune(child, target, verbose_level)
        total_subtree_error += child.error

    if verbose_level > 0:
        desc = " + ".join(f"{c.color}{c.error}(id={c.id}){rs}" for c in node.children.values())
        print(f"{node.color}Vozlišče (ID={node.id}, attr='{node.attr}'): e={errors_if_pruned}{rs}, E={desc}{rs}={total_subtree_error}", end="")

    # Odločitev: režemo ali ne
    if errors_if_pruned <= total_subtree_error:
        node.children = None
        node.attr = target
        node.split_value = None
        node.error = errors_if_pruned
        if verbose_level > 0:
            print(f"{node.color} ,e <= E ==> REŽEMO!{rs}")
    else:
        node.error = total_subtree_error
        if verbose_level > 0:
            print(f"{node.color} ,e > E ==> NE REŽEMO!{rs}")

    return node


def rep_tree_print(node, level=0, details=True):
    """Izpiše drevo z rezalnimi števci in ID oznakami."""
    col = colors[level % len(colors)]
    node.color = col
    indent = "\t" * level
    majority = node.majority_class
    correct = node.pruning_freq.get(majority, 0) if hasattr(node, "pruning_freq") else 0
    incorrect = (sum(node.pruning_freq.values()) - correct) if hasattr(node, "pruning_freq") else 0
    msg = f"{col}{indent}Vozlišče (ID={node.id}): {node.attr}{rs} -> {majority}"
    if details and hasattr(node, "pruning_freq"):
        msg += f"{col} | freq={rs}{node.pruning_freq}, {Fore.GREEN}#corrects={correct}{rs}{col}, {rs}{Fore.RED}#errors={incorrect}{rs}"
    print(msg)

    if node.children:
        for edge, child in node.children.items():
            print(f"{indent}  {col}({edge}){rs}")
            rep_tree_print(child, level + 1, details)


def rep_pruning(tree, pruning_set, target, verbose_level=0):
    """Glavna funkcija za Reduced Error Pruning."""
    if tree.root is None:
        raise ValueError("Drevo še ni zgrajeno – najprej pokliči fit().")

    # Dodeli ID-je vozliščem, da jim lahko sledimo pri izpisu
    _assign_node_ids(tree.root)

    # Vzemi vse razrede iz korena drevesa
    all_classes = list(tree.root.class_freq.keys())

    # Inicializiraj števce
    _init_pruning_counters(tree.root, all_classes)

    # Posodobi števce z rezalno množico
    rep_tree_process_pruning_set(tree.root, pruning_set, target)

    if verbose_level > 0:
        print("Števci v vozliščih po sprehodu čez rezalno množico:\n")
        rep_tree_print(tree.root)
        print("\nZačenjamo z rezanjem:\n")

    # Izvedi rezanje
    rep_tree_prune(tree.root, target, verbose_level)

    if verbose_level > 0:
        print("\nKončno drevo:\n")
        rep_tree_print(tree.root, details=False)

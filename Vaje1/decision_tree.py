import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Style, init

# --------------------------------------------
# Inicializacija barv
# --------------------------------------------
init(autoreset=False)
colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
rs = Style.RESET_ALL  # reset barv

# --------------------------------------------
# Pomožne funkcije za informacijski prispevek
# --------------------------------------------

def prob_freq(series):
    """Relativne frekvence vrednosti v seriji."""
    return series.value_counts(normalize=True, dropna=False)


def prob_cond(att, cls):
    """Pogojne verjetnosti P(class | att)."""
    if att.empty or cls.empty:
        return pd.DataFrame()
    return pd.crosstab(att, cls, normalize="index", dropna=False)


def class_entropy(cls):
    """Entropija razredov."""
    p = prob_freq(cls)
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def residual_entropy(att, cls):
    """Povprečna entropija po atributu."""
    p_cond = prob_cond(att, cls)
    if p_cond.empty:
        return 0.0
    # varna obravnava ničelnih vrednosti
    logp = np.log2(p_cond.where(p_cond > 0, 1.0))
    row_entropy = -(p_cond * logp).sum(axis=1)
    p_att = prob_freq(att).reindex(p_cond.index, fill_value=0.0)
    return float((p_att * row_entropy).sum())


def info_gain(att, cls):
    """Informacijski prispevek atributa glede na ciljno spremenljivko."""
    return class_entropy(cls) - residual_entropy(att, cls)


def find_best_threshold(att, cls):
    """Najde optimalni prag za numerični atribut glede na informacijski prispevek."""
    att_valid = att.dropna()
    cls_valid = cls.loc[att_valid.index]
    if att_valid.nunique() <= 1:
        return None, 0.0

    att_sorted = att_valid.sort_values()
    cls_sorted = cls_valid.loc[att_sorted.index]
    unique_vals = att_sorted.unique()

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    ig_values = [
        info_gain((att_sorted <= thr).astype("category"), cls_sorted)
        for thr in thresholds
    ]

    if not ig_values:
        return None, 0.0

    best_idx = int(np.argmax(ig_values))
    return float(thresholds[best_idx]), float(ig_values[best_idx])


def is_numeric_series(s):
    """Preveri, ali je serija numerična."""
    return pd.api.types.is_numeric_dtype(s)


# -------------------------------------------------------------
# Funkcija za podroben izpis izračuna informacijskega prispevka
# -------------------------------------------------------------

def explain_info_gain(att_name, att_series, cls_series, cls_name="Class", col=Style.RESET_ALL, spaces=""):
    """Podroben izračun informacijskega prispevka atributa."""
    print(col)
    print(f"\n{spaces}Atribut: {rs}{att_name}{col}")
    print(f"{spaces}─────────────────────────────────────────────")

    # 1. Relativne frekvence razredov
    p_class = prob_freq(cls_series)
    print(f"{spaces}Relativne frekvence ciljnega stolpca {rs}{cls_name} (P({cls_name})){col}:")
    print(rs)
    for c, p in p_class.items():
        print(f"{spaces}  P({cls_name}={c}) = {p:.5f}")
    print(col)

    # 2. Entropija razredov
    print(f"\n{spaces}Izračun entropije {rs}H({cls_name}){col}:")
    symbolic_terms = [f"P({cls_name}={c})*log2(P({cls_name}={c}))" for c in p_class.index]
    numeric_terms = [f"{p:.5f} * log2({p:.5f})" for p in p_class.values if p > 0]
    H_class = -(p_class[p_class > 0] * np.log2(p_class[p_class > 0])).sum()
    print(rs)
    print(f"{spaces}H({cls_name}) = -[ {' + '.join(symbolic_terms)} ] = -({ ' + '.join(numeric_terms) }) = {H_class:.5f}\n")
    print(col)

    # 3. Če je atribut numeričen
    if is_numeric_series(att_series):
        print(f"{spaces}Atribut '{rs}{att_name}{col}' je numeričen — preverjam možne prage:")
        att_valid = att_series.dropna()
        cls_valid = cls_series.loc[att_valid.index]
        att_sorted = att_valid.sort_values()
        cls_sorted = cls_valid.loc[att_sorted.index]
        unique_vals = att_sorted.unique()

        if len(unique_vals) <= 1:
            print(f"{spaces}  Vse vrednosti enake — InfoGain = 0.0\n")
            print(f"{spaces}─────────────────────────────────────────────\n")
            return 0.0, None

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        best_ig, best_thr = 0.0, None

        for thr in thresholds:
            mask = att_sorted <= thr
            left_cls = cls_sorted[mask]
            right_cls = cls_sorted[~mask]
            p_left = len(left_cls) / len(cls_sorted)
            p_right = len(right_cls) / len(cls_sorted)

            # --- Entropija leve skupine
            p_left_freq = prob_freq(left_cls)
            nonzero_left = p_left_freq[p_left_freq > 0]
            symbolic_left = [f"P({cls_name}={c}|≤{thr:.3f})*log2(P({cls_name}={c}|≤{thr:.3f}))"
                             for c in nonzero_left.index]
            numeric_left = [f"{p:.5f}*log2({p:.5f})" for p in nonzero_left.values]
            H_left = -(nonzero_left * np.log2(nonzero_left)).sum()

            # --- Entropija desne skupine
            p_right_freq = prob_freq(right_cls)
            nonzero_right = p_right_freq[p_right_freq > 0]
            symbolic_right = [f"P({cls_name}={c}|>{thr:.3f})*log2(P({cls_name}={c}|>{thr:.3f}))"
                              for c in nonzero_right.index]
            numeric_right = [f"{p:.5f}*log2({p:.5f})" for p in nonzero_right.values]
            H_right = -(nonzero_right * np.log2(nonzero_right)).sum()

            # --- Rezidualna entropija in InfoGain
            H_res = p_left * H_left + p_right * H_right
            ig = H_class - H_res

            # --- Izpis vsega skupaj
            print(rs)
            print(f"\n{spaces}{col}Prag: {rs}{thr:.5f}")
            print(f"{spaces}  P(≤ {thr:.5f}) = {p_left:.5f}, "
                  f"P(> {thr:.5f}) = {p_right:.5f}\n")

            print(f"{spaces}  H({cls_name}|≤ {thr:.3f}) = "
                  f"-[ {' + '.join(symbolic_left)} ] = "
                  f"-({' + '.join(numeric_left)} ) = {H_left:.5f}")
            print(f"{spaces}  H({cls_name}|> {thr:.3f}) = "
                  f"-[ {' + '.join(symbolic_right)} ] = "
                  f"-({' + '.join(numeric_right)} ) = {H_right:.5f}")

            print(f"{spaces}  Rezidualna entropija = "
                  f"{p_left:.5f}*{H_left:.5f} + {p_right:.5f}*{H_right:.5f} = {H_res:.5f}")
            print(f"{spaces}  InfoGain({att_name}, prag={thr:.5f}) = "
                  f"{H_class:.5f} - {H_res:.5f} = {ig:.5f}")

            if ig > best_ig:
                best_ig, best_thr = ig, thr

        print(f"\n{spaces}{col}Najboljši prag: {rs}{best_thr:.5f}")
        print(f"{spaces}InfoGain({att_name}) = {best_ig:.5f}{col}")
        print(f"{spaces}─────────────────────────────────────────────\n")
        print(rs)
        return best_ig, best_thr

    # 4. Relativne frekvence vrednosti atributa
    p_att = prob_freq(att_series)
    print(f"{spaces}Relativne frekvence vrednosti atributa {rs}{att_name} (P({att_name})){col}:")
    print(rs)
    for v, p in p_att.items():
        print(f"{spaces}  P({att_name}={v}) = {p:.5f}")
    print(col)

    # 5. Pogojne verjetnosti
    p_cond = prob_cond(att_series, cls_series)
    print(f"\n{spaces}Pogojne verjetnosti {rs}P({cls_name}|{att_name}){col}: (razredi v vrsticah, vrednosti atributa v stolpcih)")
    cond_str = p_cond.T.round(4).to_string()
    print(rs)
    print("\n".join(f"{spaces}    " + line for line in cond_str.splitlines()))
    print(col)

    # 6. Izračun rezidualne entropije
    print(f"\n{spaces}Izračun pogojne entropije {rs}H({cls_name}|{att_name}){col}:")
    total_H_cond = 0.0
    weighted_terms = []
    for att_val in p_cond.index:
        probs = p_cond.loc[att_val]
        nonzero_probs = probs[probs > 0]
        symbolic_terms = [
            f"P({cls_name}={c}|{att_name}={att_val})*log2(P({cls_name}={c}|{att_name}={att_val}))"
            for c in nonzero_probs.index
        ]
        numeric_terms = [f"{p:.5f} * log2({p:.5f})" for p in nonzero_probs.values]
        H_val = -(nonzero_probs * np.log2(nonzero_probs)).sum()
        p_a = p_att.get(att_val, 0)
        total_H_cond += p_a * H_val
        weighted_terms.append(f"{p_a:.5f} * {H_val:.5f}")

        print(rs)
        print(f"\n{spaces}  {col}Za{rs} {att_name} = '{att_val}'{col}:{rs}")
        print(f"{spaces}    H({cls_name}|{att_name}={att_val}) = -[ {' + '.join(symbolic_terms)} ] = -({ ' + '.join(numeric_terms) }) = {H_val:.5f}")

    print(f"\n{spaces}H({cls_name}|{att_name}) = Σ P({att_name}) * H({cls_name}|{att_name})")
    print(f"{spaces}                      = ({' + '.join(weighted_terms)})")
    print(f"{spaces}                      = {total_H_cond:.5f}")

    IG = H_class - total_H_cond
    print(f"\n{spaces}InfoGain({att_name}) = H({cls_name}) - H({cls_name}|{att_name}) = {H_class:.5f} - {total_H_cond:.5f} = {IG:.5f}")
    print(col)
    print(f"{spaces}─────────────────────────────────────────────\n")
    print(rs)
    return IG, None


# -----------------------------
# Razred TreeNode
# -----------------------------

class TreeNode:
    def __init__(self, attr, count, class_freq, majority_class, split_value=None, children=None):
        self.attr = attr
        self.count = count
        self.class_freq = class_freq
        self.majority_class = majority_class
        self.split_value = split_value
        self.children = children

    def is_leaf(self):
        return self.children is None

    def classify(self, sample):
        """Rekurzivna klasifikacija enega primera."""
        if self.is_leaf():
            return self.majority_class

        attr_value = sample.get(self.attr)
        if attr_value is None:
            return self.majority_class

        if self.split_value is not None:
            key = "<=" if attr_value <= self.split_value else ">"
        else:
            key = str(attr_value)

        subtree = self.children.get(key)
        return self.majority_class if subtree is None else subtree.classify(sample)

    def pretty_print(self, level=0, edge_label=None):
        indent = ""
        for l in range(level - 1):
            indent += _pick_color(l) + "│   "
        indent += (_pick_color(level - 1) + "├── " + rs if level > 0 else "")
        class_freq_text = ", ".join(f"{k}:{v}" for k, v in self.class_freq.items())

        if edge_label is not None:
            print(f"{indent}{_pick_color(level - 1) + edge_label}:{rs}")

        node_indent = ""
        for l in range(level):
            node_indent += _pick_color(l) + "│   "
        if self.is_leaf():
            print(f"{node_indent}{_pick_color(level)}Leaf → {rs}{self.majority_class} [{class_freq_text}]")
        else:
            node_label = self.attr if self.split_value is None else f"{self.attr} ≤ {self.split_value:.3f}"
            print(f"{node_indent}{_pick_color(level) + node_label + rs} [{class_freq_text}]")
            for branch_label, child in self.children.items():
                child.pretty_print(level + 1, edge_label=str(branch_label))

    def __repr__(self):
        if self.is_leaf():
            return f"<Leaf: {self.majority_class}, n={self.count}>"
        return f"<Node: {self.attr}, split={self.split_value}, n={self.count}>"


# -----------------------------
# Funkcije za izris
# -----------------------------

def _layout_tree(node, depth=0, x=0, positions=None, leaf_count=None):
    """Izračun koordinat za vsako vozlišče v drevesu."""
    if positions is None:
        positions = {}
    if leaf_count is None:
        leaf_count = [0]

    if node.is_leaf():
        positions[node] = (leaf_count[0], -depth)
        leaf_count[0] += 1
    else:
        child_positions = []
        for child in node.children.values():
            positions, leaf_count = _layout_tree(child, depth + 1, x, positions, leaf_count)
            child_positions.append(positions[child])
        x_center = np.mean([pos[0] for pos in child_positions])
        positions[node] = (x_center, -depth)

    return positions, leaf_count


def _draw_tree(ax, node, positions, parent=None, edge_label=None):
    """Rekurzivno nariše vozlišče in povezave."""
    x, y = positions[node]
    if node.is_leaf():
        boxstyle = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black")
        label = f"{node.majority_class}\n" + ",".join(f"{k}:{v}" for k, v in node.class_freq.items())
    else:
        boxstyle = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black")
        label = node.attr if node.split_value is None else f"{node.attr} ≤ {node.split_value}"
    ax.text(x, y, label, ha="center", va="center", bbox=boxstyle, fontsize=8)
    if parent is not None:
        px, py = positions[parent]
        ax.plot([px, x], [py, y], "k-")
        if edge_label:
            ax.text((px + x) / 2, (py + y) / 2 - 0.1, edge_label, fontsize=7, color="red", ha="center", va="center")
    if not node.is_leaf():
        for edge, child in node.children.items():
            _draw_tree(ax, child, positions, parent=node, edge_label=str(edge))


def _pick_color(level):
    return colors[level % len(colors)]


# -----------------------------
# DecisionTree razred
# -----------------------------

class DecisionTree:
    def __init__(self, max_depth=10**9, min_samples_for_split=2, min_samples_in_leaf=1, verbose_level=0):
        self.max_depth = max_depth
        self.min_samples_for_split = min_samples_for_split
        self.min_samples_in_leaf = min_samples_in_leaf
        self.verbose_level = verbose_level
        self.root = None

    def fit(self, data, target, attributes):
        """Zgradi odločitveno drevo iz podatkov."""
        self.root = self._build_tree(data, target, attributes, depth=1)

    def _build_tree(self, data, target, attributes, depth):
        col = _pick_color(depth - 1)
        spaces = "\t" * (depth - 1)

        # Izpisujemo samo, če je verbose_level > 0
        if self.verbose_level > 0:
            print(col)
            print(f"{spaces}Globina: {depth}")
            if self.verbose_level == 1:
                print(f"{spaces}Število učnih primerov: {rs}{len(data)}{col}")
                print(f"{spaces}Število atributov: {rs}{len(attributes)}{col}")
            elif self.verbose_level > 1:
                print(f"{spaces}Učna množica:")
                data_str = data.to_string().replace("\n", "\n" + spaces)
                print(rs + f"{spaces}{data_str}" + col)
                print()

        class_freq_series = data[target].value_counts()
        if class_freq_series.empty:
            if self.verbose_level > 0:
                print(rs)
            return TreeNode(target, 0, {}, None)

        class_freq_dict = class_freq_series.to_dict()
        majority_class = class_freq_series.idxmax()
        unique_targets = (class_freq_series > 0).sum()

        create_leaf = False
        best_split_value = None
        best_attr = None

        if unique_targets == 1:
            create_leaf = True
            if self.verbose_level > 0:
                print(f"{spaces}Vsi primeri pripadajo istemu razredu, zato naredimo list:")
        elif not attributes:
            create_leaf = True
            if self.verbose_level > 0:
                print(f"{spaces}Nimamo več atributov na voljo, zato naredimo list:")
        elif depth >= self.max_depth:
            create_leaf = True
            if self.verbose_level > 0:
                print(f"{spaces}Dosegli smo maksimalno globino drevesa, zato naredimo list:")
        elif len(data) < self.min_samples_for_split:
            create_leaf = True
            if self.verbose_level > 0:
                print(f"{spaces}Nimamo dovolj učnih primerov, zato naredimo list:")

        if not create_leaf:
            if self.verbose_level > 0:
                print(f"{spaces}Izbiramo najbolj informativen atribut za razbitje podatkov:")

            results = []
            for att in attributes:
                series = data[att]
                if self.verbose_level == 2:
                    ig, thresh = explain_info_gain(att, series, data[target], target, col, spaces)
                elif is_numeric_series(series):
                    thresh, ig = find_best_threshold(series, data[target])
                else:
                    ig = info_gain(series.astype("category"), data[target])
                    thresh = None

                results.append((att, thresh, ig))

            best_attr, best_split_value, best_gain = max(results, key=lambda x: x[2], default=(None, None, 0))

            # IZPIS: Kandidati in njihova kakovost (InfoGain)
            if self.verbose_level > 0:
                print(f"{spaces}{col}Kandidati za razbitje (atribut, prag, InfoGain):{rs}")
                for att_name, thr, ig_val in results:
                    if thr is None:
                        print(f"{spaces}  {col}- {rs}{att_name}{col}:{rs} InfoGain = {ig_val:.5f}")
                    else:
                        print(f"{spaces}  {col}- {rs}{att_name}{col} (prag = {rs}{thr:.5f}{col}):{rs} InfoGain = {ig_val:.5f}")
                print(f"{spaces}{col}Najboljši atribut: {rs}{best_attr}, InfoGain = {best_gain:.5f}{col}")
                if best_split_value is not None:
                    print(f"{spaces}  {col}→ prag = {rs}{best_split_value:.5f}")
                print()


            if best_gain == 0 or best_attr is None:
                create_leaf = True
                if self.verbose_level > 0:
                    print(f"{spaces}{col}Ni koristnih atributov, zato naredimo list:{rs}")
            else:
                if self.verbose_level > 0:
                    print(f"{spaces}{col}Naredimo notranje vozlišče z atributom {rs}{best_attr}{col}.")
                if best_split_value is not None:
                    left_sel = data[best_attr] <= best_split_value
                    right_sel = data[best_attr] > best_split_value
                    if left_sel.sum() < self.min_samples_in_leaf or right_sel.sum() < self.min_samples_in_leaf:
                        create_leaf = True
                        if self.verbose_level > 0:
                            print(f"{spaces}{col}Vsaj eno poddrevo ne bo imelo dovolj učnih primerov, zato naredimo list:{rs}")
                else:
                    level_counts = data[best_attr].value_counts()
                    if (level_counts < self.min_samples_in_leaf).any():
                        create_leaf = True
                        if self.verbose_level > 0:
                            print(f"{spaces}{col}Vsaj eno poddrevo ne bo imelo dovolj učnih primerov, zato naredimo list:{rs}")

        if create_leaf:
            node = TreeNode(target, len(data), class_freq_dict, majority_class)
            if self.verbose_level > 0:
                print(f"{spaces}{rs}{target} = {node.majority_class}{col}")
            if self.verbose_level > 1:
                print(f"{spaces}{col}Frekvenca razredov v listu:{rs}")
                freq_str = class_freq_series.to_string().replace("\n", "\n" + spaces)
                print(f"{spaces}{rs}{freq_str}{col}\n")
            return node

        node = TreeNode(best_attr, len(data), class_freq_dict, majority_class, best_split_value, {})

        if best_split_value is not None:
            if self.verbose_level > 0:
                print(f"{spaces}{col}Rekurzivno naredimo poddrevo za {_pick_color(depth)}{best_attr} <= {best_split_value:.5f}{col}:")
            node.children["<="] = self._build_tree(
                data[data[best_attr] <= best_split_value], target, attributes, depth + 1
            )
            if self.verbose_level > 0:
                print(f"{spaces}{col}Rekurzivno naredimo poddrevo za {_pick_color(depth)}{best_attr} > {best_split_value:.5f}{col}:")
            node.children[">"] = self._build_tree(
                data[data[best_attr] > best_split_value], target, attributes, depth + 1
            )
        else:
            new_attributes = [a for a in attributes if a != best_attr]
            for val, subset in data.groupby(best_attr, observed=True):
                if self.verbose_level > 0:
                    print(f"{spaces}{col}Rekurzivno naredimo poddrevo za {_pick_color(depth)}{best_attr} = {val}{col}:")
                node.children[str(val)] = self._build_tree(subset, target, new_attributes, depth + 1)

        # Izpišemo reset barv samo, če izpisujemo
        if self.verbose_level > 0:
            print(rs)
        return node

    def predict(self, test_data):
        """Vrne napovedi za vse primere."""
        if self.root is None:
            raise ValueError("Drevo še ni zgrajeno. Pokliči fit().")
        return test_data.apply(lambda row: self.root.classify(row.to_dict()), axis=1)

    def score(self, X, y):
        """Ocenjena točnost modela."""
        preds = self.predict(X)
        return float((preds == y).mean())

    def pretty_print(self):
        """Izpiše drevo v tekstovni obliki."""
        print("<Prazno drevo>" if self.root is None else "")
        if self.root:
            self.root.pretty_print()

    def plot(self):
        """Izriše drevo z matplotlib."""
        if self.root is None:
            print("Drevo je prazno.")
            return
        positions, _ = _layout_tree(self.root)
        fig, ax = plt.subplots(figsize=(max(12, len(positions) / 2), 6))
        ax.axis("off")
        _draw_tree(ax, self.root, positions)
        plt.show()

    def __repr__(self):
        return f"<DecisionTree root={self.root}>"

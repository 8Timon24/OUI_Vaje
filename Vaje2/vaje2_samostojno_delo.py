import pandas as pd
import numpy as np
from decision_tree import DecisionTree
from rep_pruning import rep_pruning

# ------------------------------------------------------------
# Branje podatkov
# ------------------------------------------------------------
train_data = pd.read_csv("census_train.csv")
test_data = pd.read_csv("census_test.csv")

target = "income"
attributes = train_data.columns.drop(target).tolist()

print(f"Oblika učnih podatkov: {train_data.shape}")
print(f"Oblika testnih podatkov: {test_data.shape}")

# ------------------------------------------------------------
# Razdelitev učne množice na TRAIN in PRUNE
# ------------------------------------------------------------
np.random.seed(0)
train_data = train_data.sample(frac=1, random_state=0).reset_index(drop=True)

n = int(0.8 * len(train_data))
train_part = train_data.iloc[:n].copy()
prune_part = train_data.iloc[n:].copy()

print(f"Učna množica: {len(train_part)}")
print(f"Rezalna množica: {len(prune_part)}")
print(f"Testna množica: {len(test_data)}")

# ------------------------------------------------------------
# Binarizacija atributov (One-Hot Encoding)
# ------------------------------------------------------------
def make_dummies(df_train, df_test, target):
    X_train = pd.get_dummies(df_train.drop(columns=[target]))
    X_test = pd.get_dummies(df_test.drop(columns=[target]))
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    train_enc = X_train.copy()
    train_enc[target] = df_train[target].values
    test_enc = X_test.copy()
    test_enc[target] = df_test[target].values
    return train_enc, test_enc

train_bin, test_bin = make_dummies(train_data, test_data, target)
train_part_bin, prune_part_bin = make_dummies(train_part, prune_part, target)

# ------------------------------------------------------------
# Pomožna funkcija za merjenje točnosti
# ------------------------------------------------------------
def evaluate(tree, test_data, target):
    obs = test_data[target]
    pred = tree.predict(test_data)
    return np.mean(obs == pred)

# ------------------------------------------------------------
# Modeli
# ------------------------------------------------------------

# --- 0. Večinski klasifikator (baseline) ---
majority_class = train_data[target].mode()[0]
pred_majority = np.full(len(test_data), majority_class)
acc0 = np.mean(pred_majority == test_data[target])
print("Večinski klasifikator", acc0)

# --- 1. Drevo: originalni atributi (brez prepruninga) ---
tree1 = DecisionTree(verbose_level=0)
tree1.fit(train_data, target, attributes)
acc1 = evaluate(tree1, test_data, target)
print("Drevo – originalni atributi", acc1)

# --- 2. Drevo: binarizirani atributi (brez prepruninga) ---
tree2 = DecisionTree(verbose_level=0)
tree2.fit(train_bin, target, [c for c in train_bin.columns if c != target])
acc2 = evaluate(tree2, test_bin, target)
print("Drevo – binarizirani atributi", acc2)

# --- 3. Drevo: originalni atributi (prepruning, max_depth=5) ---
tree3 = DecisionTree(max_depth=5, verbose_level=0)
tree3.fit(train_data, target, attributes)
acc3 = evaluate(tree3, test_data, target)
print("OG drevo s preprunningom", acc3)
# --- 4. Drevo: binarizirani atributi (prepruning, max_depth=5) ---
tree4 = DecisionTree(max_depth=5, verbose_level=0)
tree4.fit(train_bin, target, [c for c in train_bin.columns if c != target])
acc4 = evaluate(tree4, test_bin, target)
print("Drevo z binariziranimi atributi in prepruningom", acc4)
# --- 5. Drevo: originalni atributi (REP pruning) ---
tree5 = DecisionTree(verbose_level=0)
tree5.fit(train_data, target, attributes)
rep_pruning(tree5, prune_part, target, verbose_level=1)
acc5 = evaluate(tree5, test_data, target)
print("Drevo z rep pruningom", acc5)
# --- 6. Drevo: binarizirani atributi (REP pruning) ---
tree6 = DecisionTree(verbose_level=0)
tree6.fit(train_bin, target, [c for c in train_bin.columns if c != target])
rep_pruning(tree6, prune_part_bin, target, verbose_level=1)
acc6 = evaluate(tree6, test_bin, target)
print("Binarizirani atributi in rep pruning", acc6)


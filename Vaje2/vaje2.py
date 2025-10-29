import pandas as pd
import numpy as np
import copy
from decision_tree import DecisionTree
from rep_pruning import rep_pruning

# ------------------------------------------------------------
# Branje podatkov
# ------------------------------------------------------------

data = pd.read_csv("credit_data.csv")
print("Podatki:")
print(data.head())
print("\nOpis podatkov:")
print(data.describe(include='all'))

target = "credit_risk"
credit_data = pd.get_dummies(data.drop(columns=[target]))
credit_data[target] = data[target].values

print("Podatki brez večvrednostnih atributov:")
print(credit_data.head())
print("\nOpis podatkov:")
print(credit_data.describe(include='all'))


# ------------------------------------------------------------
# Naključno premešaj podatke
# ------------------------------------------------------------

np.random.seed(0)
credit_data = credit_data.sample(frac=1, random_state=0).reset_index(drop=True)

# ------------------------------------------------------------
# Razdelitev na učne, rezalne in testne množice
# ------------------------------------------------------------

train_data = credit_data.iloc[:500].copy()
prun_data  = credit_data.iloc[500:700].copy()
test_data  = credit_data.iloc[700:1000].copy()

target = "credit_risk"
attributes = [c for c in credit_data.columns if c != target]

print("\nVelikosti množic:")
print(f"Učna množica: {len(train_data)}")
print(f"Rezalna množica: {len(prun_data)}")
print(f"Testna množica: {len(test_data)}")

print("\nPorazdelitev razredov:")
print("Train:\n", train_data[target].value_counts())
print("Prune:\n", prun_data[target].value_counts())
print("Test:\n", test_data[target].value_counts())


# ------------------------------------------------------------
# Gradnja odločitvenega drevesa
# ------------------------------------------------------------

print("\nGradimo odločitveno drevo...\n")

tree = DecisionTree(verbose_level=0)
tree.fit(train_data, target, attributes)

#print("Zgrajeno drevo:")
#tree.pretty_print()

print("\nTočnost na testni množici (pred rezanjem):")
obs = test_data[target]
pred = tree.predict(test_data)
print(pd.crosstab(obs, pred, rownames=["Opazovano"], colnames=["Napovedano"]))
print(f"Točnost: {np.mean(obs == pred):.3f}")

# naredi kopijo drevesa, preden ga porežemo
orig_tree = copy.deepcopy(tree)

# ------------------------------------------------------------
# Reduced Error Pruning (REP)
# ------------------------------------------------------------

print("\n---------------------------------------")
print("  Rezanje po metodi zmanjševanja napake (REP)")
print("---------------------------------------\n")

rep_pruning(tree, prun_data, target=target, verbose_level=1)

#print("\nPrerezano drevo:")
#pruned_tree.root.pretty_print()

print("\nTočnost po rezanju:")
pred = tree.predict(test_data)
print(pd.crosstab(obs, pred, rownames=["Opazovano"], colnames=["Napovedano"]))
print(f"Točnost: {np.mean(obs == pred):.3f}")

# ------------------------------------------------------------
# Izriši originalno in porezano drevo
# ------------------------------------------------------------

orig_tree.plot()  # originalno drevo
tree.plot()       # prerezano drevo








# ------------------------------------------------------------
# Naključni gozd
# ------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ustvari in nauči model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

rf.fit(train_data[attributes], train_data[target])
pred = rf.predict(test_data[attributes])

# izpiši konfuzijsko matriko
ct = pd.crosstab(obs, pred, rownames=["Opazovano"], colnames=["Napovedano"])
print(ct)

# izračunaj točnost
acc = accuracy_score(obs, pred)
print(f"Točnost: {acc:.3f}")
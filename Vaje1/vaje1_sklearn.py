import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# =============================================================================
# 1. MNOŽICA PODATKOV: SEQUEL
# =============================================================================

# Naložimo podatke
data = pd.read_csv("sequel.csv")
print(data.describe(include="all"))

# Ločimo atribute (X) in ciljno spremenljivko (y)
X = data.drop(columns=["GetSequel"])
y = data["GetSequel"]

# -------------------------------------------------------------------------
# 1A. Kodiranje kategorij z metodo »one-hot«
# -------------------------------------------------------------------------
print("\n--- One-hot kodiranje ---")
X_encoded = pd.get_dummies(X)
print(X_encoded.head())

# Omogočimo širši izpis v konzoli
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(X_encoded.head())

# Ponastavimo prikazne nastavitve
pd.reset_option("display.max_columns")
pd.reset_option("display.width")

# Zgradimo odločitveno drevo z merilom entropije
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X_encoded, y)

# Izpis drevesa v tekstovni obliki
print(export_text(tree, feature_names=X_encoded.columns.to_list()))

# Izris drevesa
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=X_encoded.columns, class_names=tree.classes_, filled=True)
plt.title("Odločitveno drevo (one-hot kodirani atributi)")
plt.show(block=True)

# Pomembnost atributov
importances = pd.Series(tree.feature_importances_, index=X_encoded.columns)
print("\nNajpomembnejši atributi:")
print(importances.sort_values(ascending=False).head(10))


# -------------------------------------------------------------------------
# 1B. Uporaba kategoričnih atributov neposredno (brez one-hot kodiranja)
# -------------------------------------------------------------------------
print("\n--- Neposredna uporaba kategoričnih atributov ---")

X = data.drop(columns=["GetSequel"])
y = data["GetSequel"]

# Pretvorimo vse nenumerične stolpce (object, string, category) v številčne kode
for col in X.select_dtypes(include=["object", "string", "category"]).columns:
    X[col] = X[col].astype("category").cat.codes

print(X.dtypes)
print("\nAtributi po kodiranju kategorij:\n", X)

# Zgradimo in naučimo drevo
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X, y)

# Izpis drevesa
print(export_text(tree, feature_names=list(X.columns)))

# Izris drevesa
plt.figure(figsize=(10, 5))
plot_tree(tree, feature_names=X.columns, class_names=[str(c) for c in tree.classes_], filled=True)
plt.title("Odločitveno drevo (kategorične kode)")
plt.show(block=True)

# Pomembnost atributov
importances = pd.Series(tree.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))


# =============================================================================
# 2. MNOŽICA PODATKOV: IRIS
# =============================================================================

iris = pd.read_csv("iris.csv")
print(iris.describe(include="all"))

X = iris.drop(columns=["species"])
y = iris["species"]

# Zgradimo odločitveno drevo
tree_iris = DecisionTreeClassifier(criterion="entropy")
tree_iris.fit(X, y)

# Izpis drevesa
print(export_text(tree_iris, feature_names=list(X.columns)))

# Izris drevesa
plt.figure(figsize=(12, 6))
plot_tree(tree_iris, feature_names=X.columns, class_names=tree_iris.classes_, filled=True)
plt.title("Odločitveno drevo za množico IRIS")
plt.show(block=True)


# =============================================================================
# 3. MNOŽICA PODATKOV: MATH_EXAM (delitev na učno in testno množico)
# =============================================================================

math_exam = pd.read_csv("math_exam.csv")
print(math_exam.head())
print(math_exam.describe(include="all"))

X = math_exam.drop(columns=["exam"])
y = math_exam["exam"]

# Pretvorimo izbrane stolpce v kategorične, če vsebujejo številske kode
for col in ["Medu", "Fedu"]:
    if col in X.columns:
        X[col] = X[col].astype("category")

# One-hot kodiranje vseh kategoričnih atributov
X_encoded = pd.get_dummies(X)

# Razdelimo na učno in testno množico (25 % test)
train_X, test_X, train_y, test_y = train_test_split(
    X_encoded, y, test_size=0.25, random_state=0
)

# Zgradimo in naučimo model
tree_exam = DecisionTreeClassifier(criterion="entropy")
tree_exam.fit(train_X, train_y)

# Izris drevesa
plt.figure(figsize=(16, 8))
plot_tree(tree_exam, feature_names=train_X.columns, class_names=tree_exam.classes_, filled=True)
plt.title("Odločitveno drevo za množico Math Exam")
plt.show(block=True)

# Napovedi na testni množici
pred = tree_exam.predict(test_X)
print("Matrika zmot:\n", confusion_matrix(test_y, pred))
print("Točnost na testni množici:", accuracy_score(test_y, pred))

# Točnost na učni množici
pred_train = tree_exam.predict(train_X)
print("Točnost na učni množici:", accuracy_score(train_y, pred_train))

# Drevo z omejeno globino (obrezano)
tree_pruned = DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree_pruned.fit(train_X, train_y)
print("Točnost (max_depth=3):", accuracy_score(test_y, tree_pruned.predict(test_X)))

# Drevo z dodatnimi omejitvami
tree_pruned2 = DecisionTreeClassifier(criterion="entropy", min_samples_split=50, min_samples_leaf=10)
tree_pruned2.fit(train_X, train_y)
print("Točnost (min_samples_split=50, min_samples_leaf=10):", accuracy_score(test_y, tree_pruned2.predict(test_X)))

# Točnost večinskega klasifikatorja
majority_class = train_y.value_counts().idxmax()
baseline_acc = (test_y == majority_class).mean()
print("Točnost večinskega klasifikatorja:", baseline_acc)

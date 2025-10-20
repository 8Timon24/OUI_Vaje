import pandas as pd
import numpy as np
from colorama import Fore, Style
from decision_tree import DecisionTree

# -----------------------------------------------------------------------------
# Učna množica, ki vsebuje samo nominalne atribute
# -----------------------------------------------------------------------------

# Branje podatkov iz datoteke
data = pd.read_csv("sequel.csv")
print(Fore.GREEN + "Izpis celotne množice:" + Style.RESET_ALL)
print(data)
print(Fore.GREEN + "Opis atributov:" + Style.RESET_ALL)
print(data.describe(include="all"))

# Izberemo ime odvisne (ciljne) spremenljivke in imena neodvisnih spremenljivk (atributov)
target = "GetSequel"
attributes = data.columns.drop(target).tolist()

# Zgradimo odločitveno drevo
tree = DecisionTree(verbose_level=2)
tree.fit(data, target, attributes)
tree.pretty_print()
tree.plot()

# -----------------------------------------------------------------------------
# Večvrednostni atribut »Genre« predstavimo kot množico binarnih atributov
# -----------------------------------------------------------------------------

# Ustvarimo dvojiške indikatorje (0/1) za vsak žanr v stolpcu »Genre«
bin_genre = pd.get_dummies(data["Genre"], prefix="Genre")

# Originalni stolpec »Genre« nadomestimo z množico binarnih stolpcev
data_bin = pd.concat([bin_genre, data.drop(columns=["Genre"])], axis=1)

# Omogočimo širši izpis v konzoli
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(Fore.GREEN + "Učna množica, kjer je atribut »Genre« predstavljen z množico binarnih atributov:" + Style.RESET_ALL)
print(data_bin)
print(Fore.GREEN + "Opis atributov:" + Style.RESET_ALL)
print(data_bin.describe(include="all"))

# Ponastavimo prikazne nastavitve
pd.reset_option("display.max_columns")
pd.reset_option("display.width")

# Izberemo ciljno spremenljivko in atribute
target = "GetSequel"
attributes = data_bin.columns.drop(target).tolist()

# Zgradimo odločitveno drevo
tree_bin = DecisionTree(verbose_level=1)
tree_bin.fit(data_bin, target, attributes)
tree_bin.pretty_print()
tree_bin.plot()

# -----------------------------------------------------------------------------
# Učna množica, ki vsebuje nominalne in numerične atribute
# -----------------------------------------------------------------------------

iris = pd.read_csv("iris.csv")
print(Fore.GREEN + "Izpis celotne množice:" + Style.RESET_ALL)
print(iris)
print(Fore.GREEN + "Opis atributov:" + Style.RESET_ALL)
print(iris.describe(include="all"))

target = "species"
attributes = iris.columns.drop(target).tolist()

# Zgradimo odločitveno drevo
tree = DecisionTree(verbose_level=1)
tree.fit(iris, target, attributes)
tree.pretty_print()
tree.plot()

# Omejitev največje globine drevesa
tree = DecisionTree(max_depth=3, verbose_level=0)
tree.fit(iris, target, attributes)
tree.pretty_print()
tree.plot()

# Nastavitev kriterija za razbitje vozlišč
tree = DecisionTree(min_samples_for_split=20, min_samples_in_leaf=10, verbose_level=0)
tree.fit(iris, target, attributes)
tree.pretty_print()
tree.plot()

# -----------------------------------------------------------------------------
# Generiranje napovedi
# -----------------------------------------------------------------------------

math_exam = pd.read_csv("math_exam.csv")
print(math_exam.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(Fore.GREEN + "Izpis začetnih vrstic množice:" + Style.RESET_ALL)
print(math_exam.head())
print(Fore.GREEN + "Opis atributov:" + Style.RESET_ALL)
print(math_exam.describe(include="all"))

# Pretvorimo numerične vrednosti »Medu« in »Fedu« v kategorialne spremenljivke
for col in ["Medu", "Fedu"]:
    if col in math_exam.columns:
        math_exam[col] = math_exam[col].astype("category")

print(Fore.GREEN + "Opis atributov (po pretvorbi):" + Style.RESET_ALL)
print(math_exam.describe(include="all"))

# Razdelimo podatke na učno in testno množico (70 % učna, 30 % testna)
np.random.seed(100)
sel = np.random.choice(len(math_exam), size=int(len(math_exam)*0.7), replace=False)
train = math_exam.iloc[sel].copy()
test = math_exam.drop(index=math_exam.index[sel]).copy()

# Alternativna možnost razdelitve:
# train = math_exam.sample(frac=0.75, random_state=0)
# test = math_exam.drop(train.index)

target = "exam"
attributes = list(math_exam.columns.drop("exam"))

# Zgradimo odločitveno drevo
tree = DecisionTree()
tree.fit(train, target, attributes)
tree.pretty_print()
tree.plot()

# Točnost modela na testni množici
print(Fore.GREEN + "Točnost modela na testni množici:" + Style.RESET_ALL)
pred = tree.predict(test)
obs = test[target]
print(pd.crosstab(obs, pred))
tbl = pd.crosstab(obs, pred)
print((np.diag(tbl).sum()) / tbl.to_numpy().sum())
print("\n\n")

# Točnost modela na učni množici
print(Fore.GREEN + "Točnost modela na učni množici:" + Style.RESET_ALL)
predT = tree.predict(train)
print(pd.crosstab(train[target], predT))
print((predT == train[target]).mean())
print("\n\n")

# Učenje drevesa z omejitvijo globine
print(Fore.GREEN + "Učenje drevesa z omejitvijo globine (3):" + Style.RESET_ALL)
tree = DecisionTree(max_depth=3)
tree.fit(train, target, attributes)
tree.plot()
pred = tree.predict(test)
print(pd.crosstab(obs, pred))
print((pred == obs).mean())
print("\n\n")

# Učenje drevesa s spremenjenim kriterijem razbitja vozlišč
print(Fore.GREEN + "Učenje drevesa s spremenjenim kriterijem razbitja vozlišč:" + Style.RESET_ALL)
tree = DecisionTree(min_samples_for_split=50, min_samples_in_leaf=10)
tree.fit(train, target, attributes)
tree.plot()
pred = tree.predict(test)
print(pd.crosstab(obs, pred))
print((pred == obs).mean())
print("\n\n")

# Točnost večinskega klasifikatorja
print(Fore.GREEN + "Točnost večinskega klasifikatorja:" + Style.RESET_ALL)
print(train[target].value_counts())
print((obs == train[target].value_counts().idxmax()).mean())

# ==============================================
# Nalaganje in testiranje uporabljenih paketov
# ==============================================

#Nalaganje
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy

#Testiranje
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("scipy:", scipy.__version__)

# ==============================================
# Python seznam (list) – osnove
# ==============================================

velikost = [32, 45, 28, 60, 38, 52, 47, 75] # velikosti stanovanj v m2

print("\n=== Originalni seznam ===")
print(velikost)           # celoten seznam
print("Dolžina seznama:", len(velikost))   # število elementov

print("\n=== Dostop do posameznih elementov ===")
print("Tretji element (indeks 2):", velikost[2])   # 28
print("Tretji element od zadaj (indeks -3):", velikost[-3])  # 52

print("\n=== Rezanje (slicing) ===")
print("Elementi od 2. do 5. (indeksi 1:5):", velikost[1:5])     # [45, 28, 60, 38]
print("Elementi od 2. do 5. s 'korakom 2':", velikost[1:5:2])   # [45, 60]
print("Od 5. elementa naprej (indeks 4:):", velikost[4:])       # [38, 52, 47, 75]
print("Vsak 3. element, od 2. dalje (1::3):", velikost[1::3])   # [45, 60, 47]
print("Obratno od 7. do 2. elementa (6:1:-1):", velikost[6:1:-1]) # [47, 52, 38, 60, 28]



# ==============================================
# Pandas Series – osnove
# ==============================================

#import pandas as pd

s_velikost = pd.Series([32, 45, 28, 60, 38, 52, 47, 75], name="Kvadratura")

print("\n=== Originalna Pandas Series ===")
print(s_velikost)
print("Dolžina serije:", len(s_velikost))

# --------------------------
# Dostop do elementov (iloc)
# --------------------------
# iloc = dostop po poziciji (kot pri seznamih)
print("\n=== Dostop do posameznih elementov (iloc) ===")
print("Tretji element:", s_velikost.iloc[2])
print("Tretji element od zadaj:", s_velikost.iloc[-3])
print("Prvi, drugi in peti element:\n", s_velikost.iloc[[0,1,4]])

# --------------------------
# Spreminjanje elementov
# --------------------------
print("\n=== Spreminjanje elementov ===")
s_velikost.iloc[2] = 30  # spremenimo 3. element
print("Spremenjen tretji element:", s_velikost.iloc[2])

s_velikost.iloc[3] = 65  # spremenimo 4. element
print("Spremenjen četrti element:", s_velikost.iloc[3])

# --------------------------
# Rezanje (slicing)
# --------------------------
print("\n=== Rezanje (slicing) ===")
print("Elementi od 2. do 5.:")
print(s_velikost.iloc[1:5])

print("\nOd 5. elementa naprej:")
print(s_velikost.iloc[4:])

print("\nSerija v obratnem vrstnem redu:")
print(s_velikost.iloc[::-1])

# --------------------------
# Spreminjanje zaporedij (s slicingom)
# --------------------------
print("\n=== Spreminjanje zaporedij (s slicingom) ===")
s_velikost.iloc[0:3] = [35, 46, 31]  # zamenjamo prve tri
print("Po spremembi prvih treh elementov:\n", s_velikost, sep="")

# --------------------------
# Dodajanje in odstranjevanje elementov
# --------------------------
print("\n=== Dodajanje in odstranjevanje elementov ===")
# Dodajanje novega elementa (iloc ne podpira dodajanja, zato uporabimo loc)
s_velikost.loc[8] = 55
print("Po dodajanju novega elementa (indeks 8 = 55):\n", s_velikost, sep="")

# Odstranimo element (vrne novo serijo)
s_brez = s_velikost.drop(1)
print("Nova serija brez elementa z indeksom 1:\n", s_brez, sep="")

# Odstranimo več elementov
s_brez = s_velikost.drop([0, 2])
print("Nova serija brez elementov z indeksom 0 in 2:\n", s_brez, sep="")
print("POZOR: indeksi se NE resetirajo po drop()!")

# Indeks lahko resetiramo
print("\nResetiranje indeksa (razlika drop=False vs drop=True):")
print("drop=False (stari indeks postane stolpec):\n", s_brez.reset_index(drop=False), sep="")
print("drop=True (stari indeks se zavrže):\n", s_brez.reset_index(drop=True), sep="")

# --------------------------
# Metode za osnovne statistike
# --------------------------
print("\n=== Osnovne statistike ===")
print("Najmanjša kvadratura:", s_velikost.min())
print("Največja kvadratura:", s_velikost.max())
print("Povprečje:", s_velikost.mean())
print("Standardni odklon:", s_velikost.std())
print("Skupna kvadratura:", s_velikost.sum())

print("\nUrejene kvadrature (sort_values):")
print(s_velikost.sort_values())

# --------------------------
# Vrednosti in indeksi
# --------------------------
print("\n=== Vrednosti in indeksi ===")
print("Vrednosti:", s_velikost.values)   # numpy array z vrednostmi
print("Indeksi:", s_velikost.index)      # RangeIndex (0..n)

# --------------------------
# Filtriranje z logičnimi maskami
# --------------------------
print("\n=== Filtriranje z logičnimi maskami ===")

# 1. Ročna maska
maska = [False, True, False, True, False, False, False, True, True]
print("Ročna maska:", maska)
print("Izbrani elementi:\n", s_velikost[maska], sep="")

# 2. Avtomatska maska s primerjavo
pogoj = s_velikost > 50
print("\nAvtomatska maska (s_velikost > 50):")
print(pogoj)
print("Izbrani elementi:\n", s_velikost[pogoj], sep="")

# 3. Kombiniranje pogojev
pogoj_and = (s_velikost > 40) & (s_velikost < 60)
print("Elementi med 40 in 60 m2:\n", s_velikost[pogoj_and], sep="")

pogoj_or = (s_velikost < 30) | (s_velikost > 70)
print("Elementi < 30 ALI > 70 m2:\n", s_velikost[pogoj_or], sep="")

pogoj_not = ~(s_velikost > 40)
print("Elementi, ki NISO > 40 m2:\n", s_velikost[pogoj_not])

# Nadaljnja analiza filtriranih podatkov
print("Povprečje vseh stanovanj > 50 m2:", s_velikost[s_velikost > 50].mean(), sep="")

# --------------------------
# Primerjava iloc vs loc (avtomatski indeksi)
# --------------------------
print("\n=== Primerjava iloc vs loc (avtomatski indeksi) ===")

s_demo = pd.Series([32, 45, 28, 60], name="Kvadratura")
print("Originalna serija:")
print(s_demo)

print("s_demo.iloc[2]:", s_demo.iloc[2])  # tretji element po zaporedju
print("s_demo.loc[2]:", s_demo.loc[2])    # element z indeksom 2

# Rezanje
s2 = s_demo.iloc[1:]   # indeksi so zdaj 1, 2, 3
print("\nNova serija po rezanju (indeksi 1,2,3):")
print(s2)

print("s2.iloc[1]:", s2.iloc[1])  # drugi element po zaporedju = 28
print("s2.loc[2]:", s2.loc[2])    # element z indeksom 2 = 28

print("\nIzbor več elementov z loc (indeksi 1 in 2):")
print(s_demo.loc[[1, 2]])  # izberi elemente z indeksi 1 in 2



# =====================================
# Pandas Series z lastnimi indeksi
# =====================================
s_velikost = pd.Series(
    [32, 45, 28, 60],
    index=["A", "B", "C", "D"],
    name="Kvadratura"
)

print("=== Series z lastnimi indeksi ===")
print(s_velikost)

# --------------------------
# Dostop: loc vs iloc
# --------------------------
print("\n=== Dostop ===")
print("s_velikost.loc['C']:", s_velikost.loc["C"])   # dostop z indeksom -> 28
print("s_velikost.iloc[2]:", s_velikost.iloc[2])     # dostop po poziciji -> 28

# --------------------------
# Spreminjanje
# --------------------------
print("\n=== Spreminjanje ===")
s_velikost.loc["B"] = 50
print("Spremenjena kvadratura B (z loc):\n", s_velikost, sep="")

# --------------------------
# Rezanje po imenih
# --------------------------
print("\n=== Rezanje po imenih ===")
print("Od B do D:\n", s_velikost.loc["B":"D"], sep="")

# --------------------------
# Dodajanje in brisanje
# --------------------------
print("\n=== Dodajanje in odstranjevanje ===")
s_velikost.loc["E"] = 70
print("Po dodajanju E=70:\n", s_velikost, sep="")

print("Brez stanovanja C:\n", s_velikost.drop("C"), sep="")



# =====================================
# Python slovar (dict) – osnove
# =====================================
stanovanje = {
    "Kvadratura": 45,
    "Sobe": 2,
    "Razdalja_center": 3.2,
    "Prenovljeno": False,
    "Tip_ogrevanja": "Plin",
    "Cena": 120000
}

print("=== Slovar o enem stanovanju ===")
print(stanovanje)

print("\nDostop do vrednosti s ključem 'Cena':", stanovanje["Cena"])
print("Ključi (keys):", stanovanje.keys())
print("Vrednosti (values):", stanovanje.values())
print("Elementi (items):", stanovanje.items())



# =====================================
# Pandas DataFrame – več lastnosti stanovanj
# =====================================

podatki = {
    "Kvadratura": [32, 45, 28, 60, 38, 52],
    "Sobe": [1, 2, 1, 3, 2, 2],
    "Razdalja_center": [1.5, 3.2, 0.8, 5.0, 2.1, 4.3],
    "Prenovljeno": [True, False, False, True, True, False],
    "Tip_ogrevanja": ["Plin", "Elektrika", "Daljinsko", "Plin", "Drva", "Daljinsko"],
    "Cena": [85000, 120000, 75000, 200000, 99000, 160000]   # int
}

df = pd.DataFrame(podatki, index=["A", "B", "C", "D", "E", "F"])

print("=== DataFrame o stanovanjih ===")
print(df)

# --------------------------
# Osnovne poizvedbe
# --------------------------
print("\n=== Osnovne poizvedbe ===")
print("Dimenzije (vrstice, stolpci):", df.shape)
print("Tipi podatkov:\n", df.dtypes, sep="")
print("Imena stolpcev:", df.columns.tolist())
print("Imena vrstic:", df.index.tolist())
print("Prvih 3 vrstic:\n", df.head(3), sep="")
print("Zadnji 2 vrstici:\n", df.tail(2), sep="")
print(df.describe())

# --------------------------
# Dostop do stolpcev
# --------------------------
print("\n=== Dostop do stolpcev ===")
print("Stolpec Cena:\n", df["Cena"], sep="")
print("Tip objekta za stolpec Cena:", type(df["Cena"]))  # pandas.core.series.Series
print("Indeksi stolpca Cena:", df["Cena"].index)         # enaki indeksu DataFrame-a

print("Stolpec Kvadratura (kot Series):\n", df["Kvadratura"], sep="")
print("Lahko tudi s pomočjo loc:\n", df.loc[:, "Kvadratura"], sep="")
print("Lahko tudi s pomočjo iloc:\n", df.iloc[:, 0], sep="")
print("Možno je tudi tole:\n", df.Kvadratura, sep="")


print("\nUporaba metod Series na stolpcu Cena:")
print("Povprečje:", df["Cena"].mean())
print("Min:", df["Cena"].min())
print("Max:", df["Cena"].max())

# --------------------------
# Dostop do vrstic
# --------------------------
print("\n=== Dostop do vrstic ===")
print("Stanovanje C (loc):\n", df.loc["C"], sep="")
print("Možno je tudi tako:\n", df.loc["C", :], sep="")
print("Druga vrstica (iloc):\n", df.iloc[1], sep="")
print("Gre tudi tako:\n", df.iloc[1, :], sep="")

# --------------------------
# Dostop do posamezne celice
# --------------------------
print("\n=== Dostop do posamezne celice ===")
print("Cena stanovanja B (loc):", df.loc["B", "Cena"])
print("Kvadratura stanovanja D (iloc):", df.iloc[3, 0])

# --------------------------
# Dostop do blokov (rezanje)
# --------------------------
print("\n=== Dostop do blokov ===")
print("Kvadratura in Cena za stanovanja B–D:\n", df.loc["B":"D", ["Kvadratura", "Cena"]])
print("Enak rezultat z iloc:\n", df.iloc[1:4, [0, 5]])

# --------------------------
# Dodajanje novega stolpca
# --------------------------
print("\n=== Dodajanje novega stolpca ===")
df["Cena_na_m2"] = df["Cena"] / df["Kvadratura"]
print(df)

# --------------------------
# Dodajanje nove vrstice
# --------------------------
print("\n=== Dodajanje nove vrstice ===")
df.loc["G"] = [70, 3, 6.0, False, "Drva", 250000, 250000/70]
print(df)

# --------------------------
# Spreminjanje obstoječe vrstice
# --------------------------
print("\n=== Spreminjanje obstoječe vrstice ===")
df.loc["G"] = [80, 3, 6.0, False, "Plin", 250000, 250000/80]
print(df)

# --------------------------
# Odstranjevanje
# --------------------------
print("\n=== Odstranjevanje ===")
print("Brez stolpca Prenovljeno:\n", df.drop(columns=["Prenovljeno"]))
print("Brez stolpcev Kvadratura in Tip_ogrevanja:\n", df.drop(columns=["Kvadratura", "Tip_ogrevanja"]))
print("Brez prvega in tretjega stolpca:\n", df.drop(columns=df.columns[[0, 2]]))
print("Brez stanovanja C:\n", df.drop(index=["C"]))
print("Brez stanovanj A in F:\n", df.drop(index=["A", "F"]))
print("Brez tretjega in petega stanovanja:\n", df.drop(index=df.index[[2, 4]]))

# --------------------------
# Filtriranje
# --------------------------
print("\n=== Filtriranje ===")
print("Stanovanja večja od 40 m2:\n", df[df["Kvadratura"] > 40])
print("Prenovljena stanovanja bližje kot 3 km:\n",
      df[(df["Prenovljeno"] == True) & (df["Razdalja_center"] < 3)])
print("Povprečna cena stanovanj, ki imajo več kot eno sobo:", df.loc[df.Sobe > 1, "Cena"].mean())

# --------------------------
# Grupiranje
# --------------------------
print("Možne vrednosti tipa ogrevanja:", df.Tip_ogrevanja.unique())
print("\n=== Porazdelitev tipa ogrevanja ===")
print(df["Tip_ogrevanja"].value_counts())

print("\n=== Povprečna velikost stanovanja glede na tip ogrevanja ===")
print(df.groupby("Tip_ogrevanja")["Kvadratura"].mean())

print("\n=== Povprečna cena glede na tip ogrevanja in prenovljenost ===")
print(df.groupby(["Tip_ogrevanja", "Prenovljeno"])["Cena"].mean())


# Za INTERAKTIVNO delo odkomentirajte spodnje vrstice:
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

# Branje CSV datoteke
df = pd.read_csv("iris.csv")   # privzeto pričakuje "," kot ločilo
print("Prvih 5 vrstic podatkov:")
print(df.head())
print("\nDimenzije:", df.shape)
print("Stolpci:", df.columns.tolist())

plt.hist(df["sepal_length"], bins=15, edgecolor="black")
plt.xlabel("Sepal length")
plt.ylabel("Frekvenca")
plt.title("Histogram sepal length")
plt.show()

# Povprečja po vrstah
povprecja = df.groupby("species")["sepal_length"].mean()

povprecja.plot(kind="bar", color=["skyblue", "lightgreen", "salmon"])
plt.ylabel("Povprečna sepal length")
plt.title("Povprečne dolžine čašnih listov po vrstah")
plt.xticks(rotation=0)
plt.show()

df.boxplot(column="sepal_length", by="species")
plt.ylabel("Sepal length")
plt.title("Porazdelitev sepal length po vrstah")
plt.suptitle("")  # odstrani dodaten naslov
plt.show()





#
# --- NALOGE (SEZNAM) ---
#

velikost = [32, 45, 28, 60, 38, 52, 47, 75]
# NALOGA 1: Izpiši drugi element seznama
print(velikost[1])
# NALOGA 2: Izpiši prve tri elemente
print(velikost[:3])
# NALOGA 3: Izpiši vsak tretji element

# NALOGA 4: Izpiši zadnji element

# NALOGA 5: Izpiši elemente od 3. do 6.

# NALOGA 6: Izpiši seznam v obratnem vrstnem redu



#
# --- NALOGE (SERIES) ---
#

s_velikost = pd.Series([32, 45, 28, 60, 38, 52, 47, 75], name="Kvadratura")

# NALOGA 1: Izpiši prvi in zadnji element serije

# NALOGA 2: Izpiši vsak drugi element od začetka

# NALOGA 3: Zamenjaj vrednost drugega elementa z 99

# NALOGA 4: Nastavi zadnja tri elementa na vrednosti [70, 71, 72]

# NALOGA 5: Izpiši vse elemente, ki niso enaki 99

# NALOGA 6: Dodaj nov element z indeksom 99 in vrednostjo 120

# NALOGA 7: Ustvari novo serijo brez elementa z indeksom 3


s_velikost = pd.Series([32, 45, 28, 60], index=["A", "B", "C", "D"], name="Kvadratura")

# NALOGA 1: Izpiši kvadraturo stanovanja D z uporabo loc

# NALOGA 2: Izpiši kvadraturo drugega stanovanja po vrsti z uporabo iloc

# NALOGA 3: Povečaj kvadraturo stanovanja A za 5 m2

# NALOGA 4: Izpiši stanovanja od A do C

# NALOGA 5: Dodaj stanovanje F s kvadraturo 80

# NALOGA 6: Ustvari novo serijo brez stanovanja B



#
# --- NALOGE (DATAFRAME)
#

df = pd.DataFrame({
    "Kvadratura": [32, 45, 28, 60, 38, 52],
    "Sobe": [1, 2, 1, 3, 2, 2],
    "Razdalja_center": [1.5, 3.2, 0.8, 5.0, 2.1, 4.3],
    "Prenovljeno": [True, False, False, True, True, False],
    "Tip_ogrevanja": ["Plin", "Elektrika", "Daljinsko", "Plin", "Drva", "Daljinsko"],
    "Cena": [85000, 120000, 75000, 200000, 99000, 160000]}, index=["A", "B", "C", "D", "E", "F"])

# NALOGA 1: Izpiši vse cene

# NALOGA 2: Izpiši podatke o stanovanju E

# NALOGA 3: Spremeni ceno stanovanja A na 90000

# NALOGA 4: Dodaj novo stanovanje H s podatki [80, 3, 2.5, True, "Daljinsko", 210000]

# NALOGA 5: Izpiši vsa stanovanja, kjer je cena na m2 večja od 3000

# NALOGA 6: Izpiši stanovanja, ki imajo manj kot 3 sobe ALI so prenovljena

# NALOGA 7: Uredi DataFrame po stolpcu Cena padajoče

# NALOGA 8: Preštej, koliko stanovanj ima posamezno število sob

# NALOGA 9: Ustvari nov DataFrame samo s stolpci Kvadratura, Sobe in Cena

# NALOGA 10: Ustvari nov DataFrame brez stanovanja D in resetiraj indeks



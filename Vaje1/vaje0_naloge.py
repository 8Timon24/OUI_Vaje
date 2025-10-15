import pandas as pd
import matplotlib
import scipy

velikost = [32, 45, 28, 60, 38, 52, 47, 75]
# NALOGA 1: Izpiši drugi element seznama
print(velikost[1])
# NALOGA 2: Izpiši prve tri elemente
print(velikost[:3])
# NALOGA 3: Izpiši vsak tretji element
print(velikost[::3])
# NALOGA 4: Izpiši zadnji element
print(velikost[-1])
# NALOGA 5: Izpiši elemente od 3. do 6.
print(velikost[2:6])
# NALOGA 6: Izpiši seznam v obratnem vrstnem redu
print(velikost[-1::-1])

s_velikost = pd.Series([32, 45, 28, 60, 38, 52, 47, 75], name="Kvadratura")

# NALOGA 1: Izpiši prvi in zadnji element serije
print(s_velikost.iloc[[0, -1]])
# NALOGA 2: Izpiši vsak drugi element od začetka
print(s_velikost[::2])
# NALOGA 3: Zamenjaj vrednost drugega elementa z 99
s_velikost.iloc[2] = 99
print(s_velikost.iloc[2])
# NALOGA 4: Nastavi zadnja tri elementa na vrednosti [70, 71, 72]
s_velikost[-1:-4:-1] = [70, 71, 72]
print(s_velikost.iloc[-1:-4:-1])
# NALOGA 5: Izpiši vse elemente, ki niso enaki 99
print(s_velikost[s_velikost != 99])
# NALOGA 6: Dodaj nov element z indeksom 99 in vrednostjo 120
s_velikost.loc[99] = 120
# NALOGA 7: Ustvari novo serijo brez elementa z indeksom 3
s = s_velikost.drop(3)
print(s)

s_velikost = pd.Series([32, 45, 28, 60], index=["A", "B", "C", "D"], name="Kvadratura")

# NALOGA 1: Izpiši kvadraturo stanovanja D z uporabo loc
print(s_velikost.loc["D"])
# NALOGA 2: Izpiši kvadraturo drugega stanovanja po vrsti z uporabo iloc
print(s_velikost.iloc[1])
# NALOGA 3: Povečaj kvadraturo stanovanja A za 5 m2
s_velikost.loc["A"]+=5
# NALOGA 4: Izpiši stanovanja od A do C
print(s_velikost.loc["A":"C"])
# NALOGA 5: Dodaj stanovanje F s kvadraturo 80
s_velikost.loc["F"] = 80
# NALOGA 6: Ustvari novo serijo brez stanovanja B
s = s_velikost.drop("B")
print(s)


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
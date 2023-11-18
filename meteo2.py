import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



repertoire = '/home/mae/tpmaths'

noms_fichiers = []
Soleil = []
Vent = []
Température = []
Pluie = []


for fichier in os.listdir(repertoire):
    if fichier.endswith('.csv'):
        noms_fichiers.append(fichier)
        chemin_fichier = os.path.join(repertoire, fichier)
        df = pd.read_csv(chemin_fichier, skiprows=3)
        Soleil.append(df["UV_INDEX"].values)
        Pluie.append(df["PRECIP_TOTAL_DAY_MM"].values)
        Vent.append(df["WINDSPEED_MAX_KMH"].values)
        Température.append(df["MAX_TEMPERATURE_C"].values)


Soleil= pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Soleil)})
Pluie = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Pluie)})
Vent = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Vent)})
Température = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Température)})


Soleil.columns = Soleil.columns.str.replace('.csv', '')
Pluie.columns = Pluie.columns.str.replace('.csv', '')
Vent.columns = Vent.columns.str.replace('.csv', '')
Température.columns = Température.columns.str.replace('.csv', '')
# Triez les colonnes par ordre d'année
Soleil= Soleil.reindex(sorted(Soleil.columns, key=lambda x: int(x)), axis=1)
Pluie= Pluie.reindex(sorted(Pluie.columns, key=lambda x: int(x)), axis=1)
Vent= Vent.reindex(sorted(Vent.columns, key=lambda x: int(x)), axis=1)
Température= Température.reindex(sorted(Température.columns, key=lambda x: int(x)), axis=1)



soleil_array = pd.DataFrame(Soleil).to_numpy()
vent_array = pd.DataFrame(Vent).to_numpy()
pluie_array = pd.DataFrame(Pluie).to_numpy()
temperature_array = pd.DataFrame(Température).to_numpy()


correlation_uv=Soleil.corr()

correlation_precipitation = Pluie.corr()
correlation_mistral = Vent.corr()
correlation_temperature = Température.corr()



# Vous pouvez également visualiser les corrélations à l'aide de heatmap
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)


plt.xlabel('années ')
plt.ylabel('années')
plt.title("Corrélation UV")
plt.imshow(correlation_uv, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Corrélation Précipitation")
plt.xlabel('années ')
plt.ylabel('années')
plt.imshow(correlation_precipitation, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Corrélation Mistral")
plt.xlabel('années ')
plt.ylabel('années')
plt.imshow(correlation_mistral, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Corrélation Température")
plt.xlabel('années ')
plt.ylabel('années')
plt.imshow(correlation_temperature, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.tight_layout()
plt.show()


##### Matrices de transition #####

seuils_uv = [3, 5, 7]  # UV faible, modéré, élevé
seuils_precipitations = [0, 2, 7]  # Pas de pluie, pluie légère, pluie modérée
seuils_vent = [15, 25, 40]  # Vent faible, modéré, fort (en km/h)
seuils_temperature = [10, 20, 25]  # Température basse, modérée, élevée (en degrés Celsius)

# Fonction pour discrétiser les données en catégories
def discretiser_uv(valeur):
    for i, seuil in enumerate(seuils_uv):
        if valeur <= seuil:
            return f"UV faible" if i == 0 else f"UV modéré" if i == 1 else f"UV élevé"
    return f"UV élevé"  # Si la valeur est supérieure au dernier seuil

def discretiser_precipitations(valeur):
    for i, seuil in enumerate(seuils_precipitations):
        if valeur <= seuil:
            return f"Pas de pluie" if i == 0 else f"Pluie légère" if i == 1 else f"Pluie modérée"
    return f"Pluie modérée"  # Si la valeur est supérieure au dernier seuil

def discretiser_vent(valeur):
    for i, seuil in enumerate(seuils_vent):
        if valeur <= seuil:
            return f"Vent faible" if i == 0 else f"Vent modéré" if i == 1 else f"Vent fort"
    return f"Vent fort"  # Si la valeur est supérieure au dernier seuil

def discretiser_temperature(valeur):
    for i, seuil in enumerate(seuils_temperature):
        if valeur <= seuil:
            return f"Température basse" if i == 0 else f"Température modérée" if i == 1 else f"Température élevée"
    return f"Température élevée"  # Si la valeur est supérieure au dernier seuil


Soleil = Soleil.applymap(discretiser_uv)
Pluie= Pluie.applymap(discretiser_precipitations)
Vent= Vent.applymap(discretiser_vent)
Température= Température.applymap(discretiser_temperature)

print(Pluie)



import pandas as pd

# Fonction pour créer une matrice de transition
def calculer_matrice_transition(data):
    # Je convertis le tableau NumPy en un objet pandas Series
    data_series = pd.Series(data)

    # J'obtiens les catégories uniques et les trie
    categories = sorted(data_series.unique())

    # J'initialise une matrice de transition avec des catégories en tant qu'index et colonnes
    matrice_transition = pd.DataFrame(index=categories, columns=categories)

    # Je parcours chaque paire de catégories
    for i in range(len(categories)):
        for j in range(len(categories)):
            # Je calcule la probabilité de transition de la catégorie i à la catégorie j
            prob_transition = sum((data_series.shift(1) == categories[i]) & (data_series == categories[j])) / sum(data_series.shift(1) == categories[i])

            # Je stocke la probabilité dans la matrice de transition
            matrice_transition.iloc[i, j] = prob_transition

    return matrice_transition

# Calcul des matrices de transition pour chaque facteur météorologique
matrice_transition_uv = calculer_matrice_transition(Soleil.values.ravel())
matrice_transition_precipitations = calculer_matrice_transition(Pluie.values.ravel())
matrice_transition_vent = calculer_matrice_transition(Vent.values.ravel())
matrice_transition_temperature = calculer_matrice_transition(Température.values.ravel())

# J'affiche les matrices de transition
print("Matrice de transition pour UV :")
print(matrice_transition_uv)

print("\nMatrice de transition pour précipitations :")
print(matrice_transition_precipitations)

print("\nMatrice de transition pour vent :")
print(matrice_transition_vent)

print("\nMatrice de transition pour température :")
print(matrice_transition_temperature)

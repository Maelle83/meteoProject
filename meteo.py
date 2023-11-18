import os
import pandas as pd
import numpy as np

# Définissez le répertoire où se trouvent vos fichiers CSV
repertoire = '/home/mae/tpmaths'

# Liste pour stocker les DataFrames de chaque fichier CSV
dataframes = []

# Parcourez les fichiers CSV dans le répertoire
for fichier in os.listdir(repertoire):
    if fichier.endswith('.csv'):
        chemin_fichier = os.path.join(repertoire, fichier)

        # Lire le fichier CSV en sautant les 4 premières lignes
        df = pd.read_csv(chemin_fichier, skiprows=3)

        # Ajouter le DataFrame à la liste
        dataframes.append(df)

# Concaténez tous les DataFrames en une seule matrice finale (en dessous les uns des autres)
data= pd.concat(dataframes, axis=0, ignore_index=True)

# Affichez la matrice finale
print(data)
nouvelle_matrice = pd.DataFrame(index=data.index)

# Créez les catégories météorologiques
categories = ["Soleil", "Pluie", "Vent", "Température"]

# Agrégez les données pour chaque catégorie
nouvelle_matrice["Soleil"] = data["UV_INDEX"]  # Exemple : Vous pouvez agréger la température maximale pour "Soleil"
nouvelle_matrice["Pluie"] = data["PRECIP_TOTAL_DAY_MM"]  # Exemple : Vous pouvez agréger les précipitations pour "Pluie"
nouvelle_matrice["Vent"] = data["WINDSPEED_MAX_KMH"]  # Exemple : Vous pouvez agréger la vitesse du vent pour "Vent"
nouvelle_matrice["Température"] = data["MAX_TEMPERATURE_C"]  # Exemple : Vous pouvez agréger la température du matin pour "Température"

# Affichez la nouvelle matrice
print(nouvelle_matrice)

import pandas as pd

# Supposons que vous avez votre matrice de données dans un DataFrame appelé "nouvelle_matrice"
# Voici les seuils pour chaque catégorie
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

# Appliquez la fonction de discrétisation à chaque colonne de "nouvelle_matrice"
nouvelle_matrice['Soleil'] = nouvelle_matrice['Soleil'].apply(discretiser_uv)
nouvelle_matrice['Pluie'] = nouvelle_matrice['Pluie'].apply(discretiser_precipitations)
nouvelle_matrice['Vent'] = nouvelle_matrice['Vent'].apply(discretiser_vent)
nouvelle_matrice['Température'] = nouvelle_matrice['Température'].apply(discretiser_temperature)

# La matrice de données a maintenant des colonnes discrétisées en catégories

# Affichez la matrice de données discrétisée
print(nouvelle_matrice)

colonnes_parametres = ['Soleil', 'Pluie', 'Vent', 'Température']

# Dictionnaire pour stocker les matrices de probabilité initiale pour chaque paramètre
matrices_probabilite_initiale = {}

# Calculez les matrices de probabilité initiale pour chaque paramètre
for parametre in colonnes_parametres:
    # Comptez le nombre d'occurrences de chaque catégorie
    compte_categories = nouvelle_matrice[parametre].value_counts()

    # Divisez le nombre d'occurrences par le nombre total d'occurrences pour obtenir les probabilités initiales
    probabilites_initiales = compte_categories / len(nouvelle_matrice)

    # Stockez la matrice de probabilité initiale dans le dictionnaire
    matrices_probabilite_initiale[parametre] = probabilites_initiales

# Vous avez maintenant un dictionnaire de matrices de probabilité initiale pour chaque paramètre
for parametre, matrice in matrices_probabilite_initiale.items():
    print(f"Matrice de probabilité initiale pour {parametre} :")
    print(matrice)

parametre = 'pluie'  # Le paramètre pour lequel vous souhaitez calculer la matrice de transition

# Créez une copie des données discrétisées
donnees_discretes = nouvelle_matrice[parametre]

# Créez une colonne pour l'état du jour suivant en décalant les données d'un jour
donnees_discretes_suivantes = donnees_discretes.shift(-10).dropna()

# Calculez la matrice de transition d'un jour à l'autre
matrice_transition = pd.crosstab(donnees_discretes[:-1], donnees_discretes_suivantes, normalize='index')

# Affichez la matrice de transition
print("Matrice de transition d'un jour à l'autre pour", parametre, ":")
print(matrice_transition)




# Nombre de jours à prédire
jours_a_predit = 1000

# Probabilités initiales pour le paramètre
probabilites_initiales = matrices_probabilite_initiale[parametre]

# État initial (choisi en fonction des probabilités initiales)
etat_initial = np.random.choice(probabilites_initiales.index, p=probabilites_initiales)

# Effectuez la prédiction pour les 5 prochains jours
prediction = [etat_initial]
etat_actuel = etat_initial
for _ in range(jours_a_predit - 1):
    # Choisissez l'état suivant en utilisant la matrice de transition
    etat_suivant = np.random.choice(matrice_transition.columns, p=matrice_transition.loc[etat_actuel])
    prediction.append(etat_suivant)
    etat_actuel = etat_suivant

# Affichez la prédiction

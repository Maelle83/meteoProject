import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#####################################################
############# PREPARATION DES DONNEES ###############
#####################################################

repertoire = '/home/mae/tpmaths/DATA'

noms_fichiers = []
Soleil = []
Vent = []
Température = []
Pluie = []

# EXTRACTIONS DES DONNEES METEOROLOGIQUES DES 14 ANNEES
for fichier in os.listdir(repertoire):
    if fichier.endswith('.csv'):
        noms_fichiers.append(fichier)
        chemin_fichier = os.path.join(repertoire, fichier)
        df = pd.read_csv(chemin_fichier, skiprows=3)
        Soleil.append(df["UV_INDEX"].values)
        Pluie.append(df["PRECIP_TOTAL_DAY_MM"].values)
        Vent.append(df["WINDSPEED_MAX_KMH"].values)
        Température.append(df["MAX_TEMPERATURE_C"].values)

Soleil = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Soleil)})
Pluie = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Pluie)})
Vent = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Vent)})
Température = pd.DataFrame({nom_fichier: donnee for nom_fichier, donnee in zip(noms_fichiers, Température)})

# Renommer les colonnes
Soleil.columns = Soleil.columns.str.replace('.csv', '',regex=True)
Pluie.columns = Pluie.columns.str.replace('.csv', '', regex=True)
Vent.columns = Vent.columns.str.replace('.csv', '',regex=True)
Température.columns = Température.columns.str.replace('.csv', '',regex=True)

# Trier les colonnes par ordre d'année
Soleil = Soleil.reindex(sorted(Soleil.columns, key=lambda x: int(x)), axis=1)
Pluie = Pluie.reindex(sorted(Pluie.columns, key=lambda x: int(x)), axis=1)
Vent = Vent.reindex(sorted(Vent.columns, key=lambda x: int(x)), axis=1)
Température = Température.reindex(sorted(Température.columns, key=lambda x: int(x)), axis=1)

#####################################################
###########  CORRELATION ENTRE LES ANNEES ###########
#####################################################


correlation_uv = Soleil.corr()
correlation_precipitation = Pluie.corr()
correlation_mistral = Vent.corr()
correlation_temperature = Température.corr()

#ENREGISTRER LES MATRICES :

with open('/home/mae/tpmaths/RESULTS/matriceCorrelationUV.csv', 'w') as matriceCorrelationUV:
    matriceCorrelationUV.write(correlation_uv.to_csv())

with open('/home/mae/tpmaths/RESULTS/matriceCorrelationPluie.csv', 'w') as matriceCorrelationPluie:
    matriceCorrelationPluie.write(correlation_precipitation.to_csv())

with open('/home/mae/tpmaths/RESULTS/matriceCorrelationMistral.csv', 'w') as matriceCorrelationMistral:
    matriceCorrelationMistral.write(correlation_mistral.to_csv())

with open('/home/mae/tpmaths/RESULTS/matriceCorrelationTemperature.csv', 'w') as matriceCorrelationTemp:
    matriceCorrelationTemp.write(correlation_temperature.to_csv())

# Heatmap
plt.figure(figsize=(15, 10))

# Corrélation UV
plt.subplot(2, 2, 1)

plt.title("UV")
plt.imshow(correlation_uv, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_uv.columns)), correlation_uv.columns, rotation=45)
plt.yticks(range(len(correlation_uv.index)), correlation_uv.index)

# Corrélation Précipitation
plt.subplot(2, 2, 2)
plt.title("Précipitations")

plt.imshow(correlation_precipitation, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_precipitation.columns)), correlation_precipitation.columns, rotation=45)
plt.yticks(range(len(correlation_precipitation.index)), correlation_precipitation.index)

# Corrélation Mistral
plt.subplot(2, 2, 3)
plt.title("Mistral")

plt.imshow(correlation_mistral, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_mistral.columns)), correlation_mistral.columns, rotation=45)
plt.yticks(range(len(correlation_mistral.index)), correlation_mistral.index)

# Corrélation Température
plt.subplot(2, 2, 4)
plt.title("Température")

plt.imshow(correlation_temperature, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_temperature.columns)), correlation_temperature.columns, rotation=45)
plt.yticks(range(len(correlation_temperature.index)), correlation_temperature.index)

plt.tight_layout()
plt.savefig('/home/mae/tpmaths/RESULTS/heatmapAll.png')
plt.show()


# TODO: Ici, je me suis intéresser aux années fortement corrélées entre elles, avec un seuil de corrélation de 0.8 mais cette valeur est modifibale selon les besoins
# Seuil de corrélation
seuil_corr = 0.8

# Filtre les paires d'années avec corrélation supérieure à seuil_corr
significatif_uv = np.where(np.abs(correlation_uv.values) > seuil_corr)
significatif_precipitation = np.where(np.abs(correlation_precipitation.values) > seuil_corr)
significatif_mistral = np.where(np.abs(correlation_mistral.values) > seuil_corr)
significatif_temperature = np.where(np.abs(correlation_temperature.values) > seuil_corr)

# Affiche les paires d'années significatives
print("Paires d'années significatives pour la corrélation UV:")
for i, j in zip(*np.triu(significatif_uv, k=1)):
    if (i!=j):
     print(f"{correlation_uv.index[i]} - {correlation_uv.columns[j]}")

print("\nPaires d'années significatives pour la corrélation Précipitations:")
for i, j in zip(*np.triu(significatif_precipitation, k=1)):
    if (i != j):
     print(f"{correlation_precipitation.index[i]} - {correlation_precipitation.columns[j]}")

print("\nPaires d'années significatives pour la corrélation Mistral:")
for i, j in zip(*np.triu(significatif_mistral, k=1)):
    if (i != j):
     print(f"{correlation_mistral.index[i]} - {correlation_mistral.columns[j]}")

print("\nPaires d'années significatives pour la corrélation Température:")
for i, j in zip(*np.triu(significatif_temperature, k=1)):
    if (i != j):
     print(f"{correlation_temperature.index[i]} - {correlation_temperature.columns[j]}")



#######################################################
####### ANALYSE EN COMPOSANTES PRINCIPALES ############
######################################################
Température_transposed= Température.transpose()

# Centrer les données
donnees_centrees = Température_transposed - Température_transposed.mean()

# Calculer la matrice de covariance
matrice_covariance = np.cov(donnees_centrees, rowvar=False)

# Calculer les valeurs propres et les vecteurs propres
valeurs_propres, vecteurs_propres = np.linalg.eig(matrice_covariance)

# Trier les valeurs propres et les vecteurs propres
indices_tri = np.argsort(valeurs_propres)[::-1]
valeurs_propres = valeurs_propres[indices_tri]
vecteurs_propres = vecteurs_propres[:, indices_tri]

# Projeter les données sur les composantes principales
composantes_principales = np.dot(donnees_centrees, vecteurs_propres)

# Créer un DataFrame pour les composantes principales
df_pca = pd.DataFrame(data=composantes_principales, columns=[f'PC{i+1}' for i in range(composantes_principales.shape[1])])

# Ajouter les noms de lignes comme étiquettes
df_pca['Années'] = Température_transposed.index
print(df_pca)
# Visualisation en 2D avec étiquettes
plt.figure(figsize=(10, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)

# Ajouter les étiquettes pour chaque point
for i, ligne in enumerate(df_pca['Années']):
    plt.text(df_pca['PC1'][i], df_pca['PC2'][i], ligne, fontsize=8, ha='left', va='bottom')

plt.title('Analyse en composantes principales des températures du mois de Mai à Toulon')
plt.xlabel('Composante Principale 1 (PC1)')
plt.ylabel('Composante Principale 2 (PC2)')
plt.show()

#####################################################
############# MODELE PROBABILISTE     ###############
#####################################################

#### PREPARATION DES DONNEES
# Cette fois-ci, je veux toutes les données météorologiques dans le même dataframe


dataframes = []

# Parcours les fichiers CSV dans le répertoire
for fichier in os.listdir(repertoire):
    if fichier.endswith('.csv'):
        chemin_fichier = os.path.join(repertoire, fichier)

        # Lis le fichier CSV en sautant les 4 premières lignes
        df = pd.read_csv(chemin_fichier, skiprows=3)

        # Ajoute le DataFrame à la liste
        dataframes.append(df)

# Concatène tous les DataFrames en une seule matrice finale (en dessous les uns des autres)
data= pd.concat(dataframes, axis=0, ignore_index=True)

nouvelle_matrice = pd.DataFrame(index=data.index)

#### DISCRETISATION DES DONNEES

# Catégories météorologiques
categories = ["Soleil", "Pluie", "Vent", "Température"]

# Agrégez les données pour chaque catégorie
nouvelle_matrice["Soleil"] = data["UV_INDEX"]
nouvelle_matrice["Pluie"] = data["PRECIP_TOTAL_DAY_MM"]
nouvelle_matrice["Vent"] = data["WINDSPEED_MAX_KMH"]
nouvelle_matrice["Température"] = data["MAX_TEMPERATURE_C"]


# Voici les seuils pour chaque catégorie
seuils_uv = [3, 5, 7]  # UV faible, modéré, élevé
seuils_precipitations = [0, 2, 7]  # Pas de pluie, pluie légère, pluie modérée
seuils_vent = [15, 25, 40]  # Vent faible, modéré, fort (en km/h)
seuils_temperature = [10, 20, 25]  # Température basse, modérée, élevée (en degrés Celsius)

# Fonctions pour discrétiser les données en catégories :
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

# Applique la fonction de discrétisation à chaque colonne de "nouvelle_matrice" :
nouvelle_matrice['Soleil'] = nouvelle_matrice['Soleil'].apply(discretiser_uv)
nouvelle_matrice['Pluie'] = nouvelle_matrice['Pluie'].apply(discretiser_precipitations)
nouvelle_matrice['Vent'] = nouvelle_matrice['Vent'].apply(discretiser_vent)
nouvelle_matrice['Température'] = nouvelle_matrice['Température'].apply(discretiser_temperature)



colonnes_parametres = ['Soleil', 'Pluie', 'Vent', 'Température']

# Dictionnaire pour stocker les matrices de probabilité initiale pour chaque paramètre
matrices_probabilite_initiale = {}

# Calcule les matrices de probabilité initiale pour chaque paramètre
for parametre in colonnes_parametres:
    # Compte le nombre d'occurrences de chaque catégorie
    compte_categories = nouvelle_matrice[parametre].value_counts()

    # Divise le nombre d'occurrences par le nombre total d'occurrences pour obtenir les probabilités initiales
    probabilites_initiales = compte_categories / len(nouvelle_matrice)

    # Stocke la matrice de probabilité initiale dans le dictionnaire
    matrices_probabilite_initiale[parametre] = probabilites_initiales


for parametre, matrice in matrices_probabilite_initiale.items():
    print(f"Matrice de probabilité initiale pour {parametre} :")
    print(matrice)

parametre = 'Pluie'  # TODO : modifier le paramètre météorologique pour lequel on veut avoir une matrice de transition

# Crée une copie des données discrétisées
donnees_discretes = nouvelle_matrice[parametre]

# Crée une colonne pour l'état du jour suivant en décalant les données d'un jour
donnees_discretes_suivantes = donnees_discretes.shift(-10).dropna()

# Calcule la matrice de transition d'un jour à l'autre
matrice_transition = pd.crosstab(donnees_discretes[:-1], donnees_discretes_suivantes, normalize='index')

# Affiche la matrice de transition
print("Matrice de transition d'un jour à l'autre pour", parametre, ":")
print(matrice_transition)


with open('/home/mae/tpmaths/RESULTS/matriceTransitionTemperature.csv', 'w') as matriceTransitionTemp:
    matriceTransitionTemp.write(matrice_transition.to_csv())





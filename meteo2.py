import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import shapiro
import numpy as np

repertoire = '/home/mae/tpmaths'

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

# PREPARATION DES DONNEES
Soleil.columns = Soleil.columns.str.replace('.csv', '')
Pluie.columns = Pluie.columns.str.replace('.csv', '')
Vent.columns = Vent.columns.str.replace('.csv', '')
Température.columns = Température.columns.str.replace('.csv', '')

# Trie les colonnes par ordre d'année
Soleil = Soleil.reindex(sorted(Soleil.columns, key=lambda x: int(x)), axis=1)
Pluie = Pluie.reindex(sorted(Pluie.columns, key=lambda x: int(x)), axis=1)
Vent = Vent.reindex(sorted(Vent.columns, key=lambda x: int(x)), axis=1)
Température = Température.reindex(sorted(Température.columns, key=lambda x: int(x)), axis=1)

# VERIFIER LA NORMALITE DES DONNEES :

for col in Soleil.columns:
    stat, p_value=shapiro(Soleil[col])

    if (p_value > 0.05) :
        print("UV", col)
        print("warning : the sample doesn't follow the normality")


for col in Vent.columns:
    stat,p_value = shapiro(Vent[col])

    if (p_value > 0.05):
        print("Vent", col)
        print("warning : the sample doesn't follow the normality")

for col in Température.columns:
    stat,p_value = shapiro(Température[col])
    if (p_value > 0.05):
        print("température", col)
        print("warning : the sample doesn't follow the normality")
for col in Pluie.columns:
    stat,p_value = shapiro(Pluie[col])
    if (p_value > 0.05):

        print("pluie",col)
        print("warning : the sample doesn't follow the normality")

# CORRELATIONS ENTRE LES ANNEES

correlation_uv = Soleil.corr()
correlation_precipitation = Pluie.corr()
correlation_mistral = Vent.corr()
correlation_temperature = Température.corr()

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



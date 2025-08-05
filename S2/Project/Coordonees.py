import pandas as pd
import os
import requests
import pandas as pd
import time
from pyproj import Transformer

def charger_table_avec_saut_dynamique(filepath, sep='\t', encoding='latin1', header_keywords=None):
    """
    Détecte dynamiquement l’en-tête en fonction de mots-clés et charge le fichier en sautant les lignes supplémentaires.

    header_keywords : liste de mots-clés devant apparaître dans la ligne d'en-tête
    """
    if header_keywords is None:
        # Mots-clés attendus pour identifier correctement l'en-tête
        header_keywords = ['Centre']

    with open(filepath, encoding=encoding) as f:
        for i, line in enumerate(f):
            if all(keyword in line for keyword in header_keywords):
                header_line = i
                break
        else:
            raise ValueError("En-tête non trouvé dans le fichier")

    # Lecture du fichier en sautant les lignes avant l’en-tête détecté
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, skiprows=header_line, header=0)
    return df

# Fonction pour convertir les coordonnées avec pyproj
def convertir_coordonnees(x, y):
    if pd.isna(x) or pd.isna(y):
        return pd.NA, pd.NA
    lon, lat = transformer.transform(x, y)
    return lat, lon

transformer = Transformer.from_crs("EPSG:27572", "EPSG:4326", always_xy=True)

# Liste des fichiers
noms_fichiers = ["IPT_BN", "IPT_HN", "IAT_BN", "IAT_HN", "DEIE_BN", "DEIE_HN"]
# noms_fichiers = ["IPT_BN", "IPT_HN", "IAT_BN", "IAT_HN"]

# Liste pour stocker les DataFrames
dataframes = []

for nom in noms_fichiers:
    chemin_fichier = f"{nom}.txt"  # Supposé être dans le même répertoire

    try:
        df = charger_table_avec_saut_dynamique(chemin_fichier)

        # Renommer colonne pour fichiers IPT
        if nom in ["IPT_BN", "IPT_HN"] and "Code Gdo Ipt" in df.columns:
            df = df.rename(columns={"Code Gdo Ipt": "Code Gdo"})

        # Renommer colonne pour fichiers IAT
        if nom in ["IAT_BN", "IAT_HN"] and "Code Gdo Iat" in df.columns:
            df = df.rename(columns={"Code Gdo Iat": "Code Gdo"})

        # Renommer colonne pour fichiers DEIE
        if nom in ["DEIE_BN", "DEIE_HN"] and "Code GDO Pdl" in df.columns:
            df = df.rename(columns={"Code GDO Pdl": "Code Gdo"})

        # Ajouter les colonnes GeoX et GeoY si elles n'existent pas
        for col in ["GeoX", "GeoY"]:
            if col not in df.columns:
                df[col] = pd.NA

        # Filtrer uniquement les colonnes nécessaires
        df_filtre = df[["Code Gdo", "GeoX", "GeoY"]].copy()

        # Ajouter le nom de la source
        df_filtre["Source"] = nom

        # Convertir GeoX et GeoY en float (en traitant les valeurs manquantes)
        df_filtre["GeoX"] = pd.to_numeric(df_filtre["GeoX"], errors='coerce')
        df_filtre["GeoY"] = pd.to_numeric(df_filtre["GeoY"], errors='coerce')

        # Appliquer la conversion
        converti = df_filtre.apply(lambda row: convertir_coordonnees(row["GeoX"], row["GeoY"]), axis=1)
        df_filtre["Latitude"] = converti.apply(lambda x: x[0])
        df_filtre["Longitude"] = converti.apply(lambda x: x[1])

        dataframes.append(df_filtre)

    except FileNotFoundError:
        print(f"Fichier {chemin_fichier} non trouvé.")
    except Exception as e:
        print(f"Erreur lors du traitement de {chemin_fichier} : {e}")

# Concaténer tous les DataFrames
df_final = pd.concat(dataframes, ignore_index=True)
df_final_unique = df_final.drop_duplicates(subset="Code Gdo", keep='first')

# # Afficher les premières lignes
# print(df_final_unique.head())

# print("\n{} équipements".format(df_final_unique["Code Gdo"].count()))
# print("Dont {} équipements avec coordonées \n".format(df_final_unique["Latitude"].count()))

# Importer le fichier PS_CARR_OMT.csv
ps_df = pd.read_csv("PS_CARR_OMT.csv", dtype=str)

# Convertir les colonnes de latitude et longitude
ps_df["PS_LATITUDE"] = pd.to_numeric(ps_df["PS_LATITUDE"], errors='coerce')
ps_df["PS_LONGITUDE"] = pd.to_numeric(ps_df["PS_LONGITUDE"], errors='coerce')

# Renommer les colonnes pour faciliter la fusion
ps_df = ps_df.rename(columns={
    "PS_CODE_GDO": "Code Gdo",
    "PS_LATITUDE": "PS_Latitude",
    "PS_LONGITUDE": "PS_Longitude"
})

# Supprimer les doublons
ps_df_unique = ps_df.drop_duplicates(subset="Code Gdo", keep='first')

# Fusionner pour ajouter les coordonnées alternatives
df_merge = df_final_unique.merge(ps_df_unique[["Code Gdo", "PS_Latitude", "PS_Longitude"]], on="Code Gdo", how="left")

# Fonction pour remplacer les valeurs manquantes
def remplir_coordonnees(row):
    lat = row["Latitude"]
    lon = row["Longitude"]
    if pd.isna(lat) or pd.isna(lon):
        if not pd.isna(row["PS_Latitude"]) and not pd.isna(row["PS_Longitude"]):
            return pd.Series([row["PS_Latitude"], row["PS_Longitude"]])
    return pd.Series([lat, lon])

# Appliquer le remplacement
df_merge[["Latitude", "Longitude"]] = df_merge.apply(remplir_coordonnees, axis=1)

# Supprimer les colonnes intermédiaires
df_merge = df_merge.drop(columns=["PS_Latitude", "PS_Longitude"])

# Mettre à jour le DataFrame final
df_final = df_merge

# print(df_final.head())

# print("\n{} équipements".format(df_final["Code Gdo"].count()))
# print("Dont {} équipements avec coordonées \n".format(df_final["Latitude"].count()))

def geocodage_inverse(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    url = f'https://api-adresse.data.gouv.fr/reverse/?lat={lat}&lon={lon}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features')
            if features:
                return features[0]['properties']['label']
    except requests.RequestException:
        return None
    return None

adresses = []
compteur = 0
tot = df_final.count()

for idx, row in df_final.iterrows():
    adresse = geocodage_inverse(row['Latitude'], row['Longitude'])
    adresses.append(adresse)

    time.sleep(0.2)  # respecter la limite de l'API

    if compteur % 250 == 0 and compteur > 0:
        print(f"{compteur}/{tot} enregistrements traités")
    compteur += 1

df_final['Adresse'] = adresses

# Enregistrer le résultat
df_final.to_csv('Coordonees.csv', encoding='utf-8-sig', index=False)

# print("Processus terminé !")

# print("\n{} équipements".format(df_final["Code Gdo"].count()))
# print("Dont {} équipements avec adresse \n".format(df_final["Adresse"].count()))
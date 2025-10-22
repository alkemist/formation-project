from datetime import datetime

import pandas as pd
import numpy as np
import os
import sys

def distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance de Haversine (distance du grand cercle) entre deux points
    définis par leurs coordonnées de latitude et longitude (en degrés).
    Retourne la distance en kilomètres (km).
    """
    R_TERRE_KM = 6371

    # Convertir les degrés en radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Différences de coordonnées
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Formule de Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return round(R_TERRE_KM * c)

DEPT = None
INPUT_FILE = 'data/formated/cities-france.csv'
OUTPUT_FILE = f"data/formated/city_pairs{''.join(['_', DEPT]) if DEPT else ''}.csv"

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if os.path.exists(OUTPUT_FILE):
    sys.exit("File '{}' exists".format(OUTPUT_FILE))

df_cities = pd.read_csv(
    INPUT_FILE,
    index_col=0,
    dtype={0: str, 'dep_code': str, 'latitude_mairie': float, 'longitude_mairie': float},
)

df_cities_filtered = df_cities[df_cities['dep_code'] == DEPT] if DEPT else df_cities

i = 0
total = df_cities_filtered.shape[0]

for city in df_cities_filtered.index:
    i += 1
    print(f"Traitement {round(i * 100 / total, 3)} % : {i} / {total} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({city})")

    df_cities_a = df_cities\
        .loc[city:city]\
        .reset_index()\
        .loc[:, ['code_insee', 'latitude_mairie', 'longitude_mairie']]\
        .rename(
            columns={'code_insee': 'city_a', 'latitude_mairie': 'lat_a', 'longitude_mairie': 'lon_a'}
        )
    df_cities_a['key'] = 1

    df_cities_b = df_cities.reset_index()\
        .loc[:, ['code_insee', 'latitude_mairie', 'longitude_mairie']]\
        .rename(
            columns={'code_insee': 'city_b', 'latitude_mairie': 'lat_b', 'longitude_mairie': 'lon_b'}
        )
    df_cities_b['key'] = 1

    df_pairs = pd.merge(df_cities_a, df_cities_b, on='key').drop('key', axis=1)

    df_pairs['distance'] = distance(
        df_pairs['lat_a'], df_pairs['lon_a'],
        df_pairs['lat_b'], df_pairs['lon_b']
    )

    df_pairs.loc[:, ['city_a', 'city_b', 'distance']].to_csv(
        OUTPUT_FILE,
        index=False,
        mode='a',
        header=not os.path.exists(OUTPUT_FILE),
    )


print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

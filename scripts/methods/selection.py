from datetime import datetime
import pandas as pd

from formation.helpers.old.map import calcul_distances

CITY_COUNT = 255
INPUT_DIR = 'data/formated/'
OUTPUT_DIR = 'data/formated/layers/'

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df_cities = pd.read_csv(INPUT_DIR + 'cities-france.csv',
    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
).rename(columns={'code_insee': 'index', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'}) \
    .set_index('index') \
    .loc[:, ['lat', 'lng']]

df_points_coords, df_points_distances = calcul_distances(df_cities.sample(CITY_COUNT))

df_points_coords.to_csv(OUTPUT_DIR + 'layer_1_points.csv', index=False)
df_points_distances.to_csv(OUTPUT_DIR + 'layer_1_distances.csv')

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
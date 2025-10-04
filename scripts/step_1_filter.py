from datetime import datetime

import pandas as pd

INPUT_FILE = 'data/brut/communes-france-2025.csv'
OUTPUT_FILE = 'data/formated/cities-france.csv'

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df_cities = pd.read_csv(INPUT_FILE, index_col=0)

df_cities['dep_code'] = df_cities['dep_code'].astype(str)
df_cities['code_insee'] = df_cities['code_insee'].astype(str)

df_cities_filtered = df_cities.loc[
    ((df_cities['dep_code'] >= '01') & (df_cities['dep_code'] <= '19'))
    | ((df_cities['dep_code'] >= '21') & (df_cities['dep_code'] <= '29'))
    | ((df_cities['dep_code'] >= '30') & (df_cities['dep_code'] <= '95'))
    , [
        'code_insee',
        'nom_standard',
        'dep_code',
        'population',
        'superficie_km2',
        'densite',
        'latitude_mairie',
        'longitude_mairie',
    ]
].sort_values(['code_insee'], ascending=True).reset_index(drop=True)

#df_cities_filtered['calculated'] = False

df_cities_filtered.set_index('code_insee').to_csv(OUTPUT_FILE)

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



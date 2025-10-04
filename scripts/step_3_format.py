import pandas as pd
import os

INPUT_FILE = 'data/formated/city_pairs.csv'
OUTPUT_FILE = 'data/formated/city_pairs_formated.csv'
CHUNK_SIZE = 100000000

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

header_written = False

chunk_iterator = pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype={'city_a': str, 'city_b': str, 'distance': float},
)

with_header = True

# 2. Boucler sur chaque morceau
for df_cities in chunk_iterator:
    #df_cities = df_cities.loc[
    #    (
    #            ((df_cities['city_a'] >= '01') & (df_cities['city_a'] <= '19'))
    #            | ((df_cities['city_a'] >= '21') & (df_cities['city_a'] <= '29'))
    #            | ((df_cities['city_a'] >= '30') & (df_cities['city_a'] <= '95'))
    #    ) & (
    #            ((df_cities['city_b'] >= '01') & (df_cities['city_b'] <= '19'))
    #            | ((df_cities['city_b'] >= '21') & (df_cities['city_b'] <= '29'))
    #            | ((df_cities['city_b'] >= '30') & (df_cities['city_b'] <= '95'))
    #    )
    #]

    # df_cities['distance'] = round(df_cities['distance'])
    df_cities['distance'] = df_cities['distance'].astype(int)

    df_cities.to_csv(OUTPUT_FILE, index=False,  mode='a', header=with_header)

    with_header = False

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
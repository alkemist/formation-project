from datetime import datetime

import pandas as pd
import os

INPUT_FILE = 'data/formated/cities-france.csv'
OUTPUT_FILE = 'data/formated/city_distances.csv'

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df_cities = pd.read_csv(
    INPUT_FILE,
    dtype={'dep_code': str}
)

df_distances = pd.DataFrame()

for dep_code in df_cities["dep_code"].unique():
    print(f"Traitement {dep_code} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_distances_dept = pd.read_csv(f'data/formated/city_distances/city_distances_{dep_code}.csv')

    df_distances = pd.concat([df_distances, df_distances_dept], ignore_index=True)

print(f"-- Save at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df_distances.to_csv(OUTPUT_FILE, index=False)

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
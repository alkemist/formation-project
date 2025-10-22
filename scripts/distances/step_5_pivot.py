from datetime import datetime

import pandas as pd
import os

INPUT_FILE = 'data/formated/cities-france.csv'
OUTPUT_DIR = 'data/formated/city_distances'

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df_cities = pd.read_csv(
    INPUT_FILE,
    dtype={'dep_code': str}
)

for dep_code in df_cities["dep_code"].unique():
    file_name = os.path.join(OUTPUT_DIR, f'city_distances_{dep_code}.csv')
    exist = os.path.exists(file_name)

    print(f"Traitement {dep_code} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {'exist' if exist else 'not exist'}")

    if not exist:
        df_pairs_dept = pd.read_csv(f'data/formated/city_pairs/city_pairs_{dep_code}.csv')

        df_pairs_dept.pivot(index='city_a', columns='city_b', values='distance')\
            .to_csv(file_name)

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
from datetime import datetime

import pandas as pd
import os

INPUT_FILE = 'data/formated/city_pairs.csv'
OUTPUT_DIR = 'data/formated/city_pairs'
CHUNK_SIZE = 100000000

total_current = 0
total_count = 34447 ** 2

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

chunk_iterator = pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype={'city_a': str, 'city_b': str, 'distance': int},
)

for df_chunk in chunk_iterator:
    df_count = df_chunk.shape[0]
    total_current = total_current + df_count

    print(f"Traitement {round(total_current * 100 / total_count, 3)} % : {total_current} / {total_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_chunk['departement'] = df_chunk['city_a'].apply(lambda x: x[:2])

    for dep_code, df_dept in df_chunk.groupby('departement'):
        file_name = os.path.join(OUTPUT_DIR, f'city_pairs_{dep_code}.csv')

        df_dept.drop(columns=['departement']).to_csv(
            file_name,
            index=False,
            mode='a',
            header=not os.path.exists(file_name),
        )

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
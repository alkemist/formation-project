import pandas as pd

DF_CITIES = pd.read_csv(
    'data/formated/cities-france.csv',
    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
).rename(columns={'code_insee': 'index', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'}) \
    .set_index('index') \
    .loc[:, ['name', 'lat', 'lng']]
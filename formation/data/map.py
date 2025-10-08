import pandas as pd
import plotly.express as px

CENTER_START = [46.227638, 2.213749]
MIN_LAT = 42
MAX_LAT = 51.5
MIN_LNG = -5
MAX_LNG = 9
ZOOM_START = 6

DF_CITIES = pd.read_csv(
    'data/formated/cities-france.csv',
    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
).rename(columns={'code_insee': 'index', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'}) \
    .set_index('index') \
    .loc[:, ['name', 'lat', 'lng']]

COLORS = px.colors.qualitative.Vivid
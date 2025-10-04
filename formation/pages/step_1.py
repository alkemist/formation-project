import streamlit as st
import folium
import random
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

from formation.data.pages import list_pages

st.title(list_pages['step_1']['title'])
st.set_page_config(
    page_title=list_pages['step_1']['title'],
    page_icon=list_pages['step_1']['icon'],
    layout="wide",
)

CENTER_START = [46.227638, 2.213749]
ZOOM_START = 6

if "center" not in st.session_state:
    st.session_state["center"] = CENTER_START
if "zoom" not in st.session_state:
    st.session_state["zoom"] = ZOOM_START
if "markers" not in st.session_state:
    st.session_state["markers"] = []
if "lines" not in st.session_state:
    st.session_state["lines"] = []
if "cities" not in st.session_state:
    st.session_state["cities"] = None

min_lat = 42
max_lat = 51.5
min_lon = -5
max_lon = 9

df_cities = pd.read_csv(
    'data/formated/cities-france.csv',
    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
).rename(columns={'code_insee': 'code', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'})\
    .set_index('code')\
    .loc[:, ['name', 'lat', 'lng']]

left, middle, right = st.columns([1, 2, 2])

def reset():
    st.session_state["markers"] = []
    st.session_state["lines"] = []

def center_map():
    st.session_state["zoom"] = ZOOM_START + random.uniform(0.00001, 0.00009)
    st.session_state["center"] = [
        CENTER_START[0] + random.uniform(0.00001, 0.00009),
        CENTER_START[1] + random.uniform(0.00001, 0.00009),
    ]

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

def random_cities(n: int):
    df_cities_random = df_cities.sample(n)
    df_cities_random['next'] = None
    df_cities_random['next'] = df_cities_random['next'].astype(str)

    city_codes_random = df_cities_random.index.unique()
    distances = {}
    city_added = [city_codes_random[0]]

    for city_code_random in city_codes_random:
        distances[city_code_random] = {}

        for city_code in list(
                filter(
                    lambda v: v != city_code_random,
                    city_codes_random
                )
        ):
            distances[city_code_random][city_code] = distance(
                df_cities_random.loc[city_code_random:city_code_random, 'lat'].values[0],
                df_cities_random.loc[city_code_random:city_code_random, 'lng'].values[0],
                df_cities_random.loc[city_code:city_code, 'lat'].values[0],
                df_cities_random.loc[city_code:city_code, 'lng'].values[0],
            )

    index = 0
    while True:
        city_code = city_added[-1]

        cities_not_added = {
            k: v
            for k, v in distances[city_code].items()
            if k not in city_added
        }

        if len(cities_not_added) > 0:
            city_next = min(cities_not_added, key=lambda k: cities_not_added[k])

            city_added.append(city_next)

            df_cities_random.loc[city_code, 'next'] = city_next
            df_cities_random.loc[city_code, 'index'] = index

            index += 1
        else:
            df_cities_random.loc[city_code, 'next'] = city_added[0]
            df_cities_random.loc[city_code, 'index'] = index
            break

    return df_cities_random.sort_values(by='index')

with left:
    st.write('Nombre de villes :')

    form_left, form_right = st.columns([4, 4], vertical_alignment='center')

    city_count = form_left.slider("Nombre de ville :", value=3, min_value=3, max_value=10, step=1, label_visibility='collapsed')

    b1, b2, b3 = form_right.columns([2, 1, 1], vertical_alignment='center')

    if b1.button("Calcul", type="primary", use_container_width=True):
        reset()
        center_map()

        st.session_state["cities"] = random_cities(city_count)

        for index, row in st.session_state["cities"].iterrows():
            st.session_state["markers"].append(
                folium.Marker(
                    location=[
                        row['lat'],
                        row['lng'],
                    ],
                    tooltip=row['name'],
                    #popup=row['nom_standard'],
                ),
            )

            city_next = st.session_state["cities"].loc[row['next']:row['next']]

            st.session_state["lines"].append(
                folium.PolyLine(
                    locations=[
                        [row['lat'], row['lng']],
                        [city_next['lat'].values[0], city_next['lng'].values[0]],
                    ],
                    color="#000000",
                    weight=3,
                )
            )

        #    folium.Circle(
        #        location=CENTER_START,
        #        radius=100,
        #        color="black",
        #        weight=1,
        #        fill_opacity=0.6,
        #        opacity=1,
        #        fill_color="green",
        #        fill=False,
        #        popup="circle",
        #        tooltip="I am in meters",
        #    )
        #]

    if b2.button(label="", type="tertiary", icon=":material/replay:"):
        reset()

    if b3.button(label="", type="tertiary", icon=":material/home:"):
        center_map()

with right:
    if st.session_state["cities"] is not None:
        st.dataframe(st.session_state["cities"])

with middle:
    m = folium.Map(
        location=CENTER_START,
        zoom_start=ZOOM_START,
        min_zoom=6,
        max_bounds=True,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )

    fg = folium.FeatureGroup(name="Markers")

    folium.PolyLine(
        locations=[
            [min_lat, min_lon],
            [max_lat, min_lon],
            [max_lat, max_lon],
            [min_lat, max_lon],
            [min_lat, min_lon],
        ],
        color="#FF0000",
        weight=2,
        tooltip="From Boston to San Francisco",
    ).add_to(fg)

    for marker in st.session_state["markers"]:
        fg.add_child(marker)

    for line in st.session_state["lines"]:
        fg.add_child(line)

    st_folium(
        m,
        center=st.session_state["center"],
        zoom=st.session_state["zoom"],
        width='100%',
        feature_group_to_add=fg,
        returned_objects=['center', 'zoom'],
        key="map",
    )
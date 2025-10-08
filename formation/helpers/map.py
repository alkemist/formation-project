import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS
import folium
import random
from shapely import MultiPoint, centroid
from sklearn.preprocessing import StandardScaler
from streamlit.delta_generator import DeltaGenerator
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from formation.data.map import DF_CITIES, COLORS, CENTER_START, ZOOM_START, MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG


def map_init():
    if "center" not in st.session_state:
        st.session_state["center"] = CENTER_START
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = ZOOM_START
    if "city_count" not in st.session_state:
        st.session_state["city_count"] = None
    if "markers_city" not in st.session_state:
        st.session_state["markers_city"] = []
    if "markers_city" not in st.session_state:
        st.session_state["markers_lines"] = []
    if "distances" not in st.session_state:
        st.session_state["distances"] = None

    if "city_tours" not in st.session_state:
        st.session_state["city_tours"] = None

    if "cities" not in st.session_state:
        st.session_state["cities"] = None
    if "markers_center" not in st.session_state:
        st.session_state["markers_center"] = []
    if "centers" not in st.session_state:
        st.session_state["centers"] = None
    if "clustering_algo" not in st.session_state:
        st.session_state["clustering_algo"] = None
    if "n_clusters_kmeans" not in st.session_state:
        st.session_state["n_clusters_kmeans"] = 1
    if "n_clusters_gaussian" not in st.session_state:
        st.session_state["n_clusters_gaussian"] = 1

def map_reset():
    st.session_state["city_count"] = None
    st.session_state["markers_city"] = []
    st.session_state["markers_center"] = []
    st.session_state["markers_lines"] = []
    st.session_state["distances"] = None

    st.session_state["city_tours"] = None

    st.session_state["cities"] = None
    st.session_state["centers"] = None
    st.session_state["n_clusters_kmeans"] = 1
    st.session_state["n_clusters_gaussian"] = 1

def map_center():
    st.session_state["zoom"] = ZOOM_START + random.uniform(0.00001, 0.00009)
    st.session_state["center"] = [
        CENTER_START[0] + random.uniform(0.00001, 0.00009),
        CENTER_START[1] + random.uniform(0.00001, 0.00009),
    ]

def calcul_distance(lat1, lon1, lat2, lon2):
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
    df_cities_random = DF_CITIES.sample(n)

    city_codes_random = df_cities_random.index.unique()

    df_pairs = pd.DataFrame({
        'city_a': [],
        'city_b': [],
        'distance': []
    })

    for city_a in city_codes_random:
        for city_b in city_codes_random:
            if city_a != city_b:

                df_pairs = pd.concat([
                    df_pairs,
                    pd.DataFrame({
                        'city_a': [city_a],
                        'city_b': [city_b],
                        'distance': [
                            calcul_distance(
                                df_cities_random.loc[city_a:city_a, 'lat'].values[0],
                                df_cities_random.loc[city_a:city_a, 'lng'].values[0],
                                df_cities_random.loc[city_b:city_b, 'lat'].values[0],
                                df_cities_random.loc[city_b:city_b, 'lng'].values[0],
                            )
                        ]
                    })
                ], ignore_index=True)

    df_distances = df_pairs.rename(columns={'city_a': 'city'})\
        .pivot(index='city', columns='city_b', values='distance')\
        .fillna(0)

    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    points = model.fit_transform(
        df_distances.to_numpy(),
    )

    df_cities = pd.DataFrame(
        points,
        columns=['x', 'y']
    )

    df_cities['cluster'] = '0'
    df_cities['index'] = df_distances.index

    df_cities = pd.merge(
        df_cities,
        df_cities_random,
        left_on='code',
        right_index=True,
    )

    return df_cities, df_distances

def calcul_distances(df: pd.DataFrame):
    indexes = df.index.unique()

    df_pairs = pd.DataFrame({
        'point_a': [],
        'point_b': [],
        'distance': []
    })

    for point_a in indexes:
        for point_b in indexes:
            if point_a != point_b:
                df_pairs = pd.concat([
                    df_pairs,
                    pd.DataFrame({
                        'point_a': [point_a],
                        'point_b': [point_b],
                        'distance': [
                            calcul_distance(
                                df.loc[point_a:point_a, 'lat'].values[0],
                                df.loc[point_a:point_a, 'lng'].values[0],
                                df.loc[point_b:point_b, 'lat'].values[0],
                                df.loc[point_b:point_b, 'lng'].values[0],
                            )
                        ]
                    })
                ], ignore_index=True)

    df_distances = df_pairs.rename(columns={'point_a': 'index'}) \
        .pivot(index='index', columns='point_b', values='distance') \
        .fillna(0)

    if df_distances.shape[0] > 1:
        model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
        points = model.fit_transform(
            df_distances.to_numpy(),
        )

        df_points = pd.DataFrame(
            points,
            index=df_distances.index,
            columns=['x', 'y']
        )

        if 'cluster' not in df.columns:
            df_points['cluster'] = '0'
        else:
            df_points['cluster'] = df_distances.index

        return pd.merge(
            df,
            df_points,
            left_index=True,
            right_index=True,
        ), df_distances

    return df, df_distances

def calcul_min_path(df_points: pd.DataFrame, df_distances: pd.DataFrame):
    indexes = df_points.index.unique()

    index_started = []
    points_tours = {}

    while True:
        index_not_started = list(
            filter(
                lambda v: v not in index_started,
                indexes
            )
        )

        if len(index_not_started) > 0:
            point_start = index_not_started[0]
            index_started.append(point_start)

            points_added = [point_start]
            distance_total = 0
            while True:
                city_code = points_added[-1]
                distances = df_distances[city_code].to_dict()

                index_not_added = {
                    k: v
                    for k, v in distances.items()
                    if k not in points_added
                }

                if len(index_not_added) > 0:
                    index_next = min(index_not_added, key=lambda k: index_not_added[k])
                    distance_total += min(index_not_added.values())

                    points_added.append(index_next)
                else:
                    break

            points_tours[point_start] = {
                'distance': distance_total,
                'indexes': points_added
            }

        else:
            break

    best_tours = min(points_tours.values(), key=lambda obj: obj["distance"])['indexes']

    df_points['order'] = df_points.index.map({
        code: i
        for i, code in enumerate(best_tours)
    })

    df_points['index'] = df_points.sort_values(by='order').index

    df_points['next'] = df_points['index'].shift(periods=-1, fill_value='')
    df_points['next'] = df_points['next'].astype(str)
    df_points.loc[df_points.index[-1], 'next'] = df_points.iloc[0]['index']
    df_points["distance"] = df_points.apply(
        lambda r: df_distances.loc[r["index"], r["next"]] if r["next"] else 0,
        axis=1
    )

    return df_points.set_index('index').drop(columns=['order', 'cluster', 'x', 'y'])

def draw_markers(map_index: int, df_points: pd.DataFrame):
    st.session_state[f"markers_point_{map_index}"] = []

    for index, row in df_points.iterrows():
        st.session_state[f"markers_point_{map_index}"].append(
            folium.CircleMarker(
                location=[
                    row['lat'],
                    row['lng'],
                ],
                tooltip=f"{index} ({row['cluster']})",
                fill=True,
                color=COLORS[int(row['cluster'])],
                weight=0,
                fill_opacity=0.6,
                radius=15,
            )
        )

        if 'next' in df_points.columns:
            city_next = df_points.loc[row['next']:row['next']]

            st.session_state[f"markers_line_{map_index}"].append(
                folium.PolyLine(
                    locations=[
                        [row['lat'], row['lng']],
                        [city_next['lat'].values[0], city_next['lng'].values[0]],
                    ],
                    color="#000000",
                    weight=3,
                )
            )

def calcul_centroids(df_points: pd.DataFrame):
    clusters = df_points['cluster'].unique()
    k = len(clusters)

    df_centroids = pd.DataFrame({
        'lat': [0] * k,
        'lng': [0] * k
    }, index=clusters)

    df_centroids['lat'] = df_centroids['lat'].astype(float)
    df_centroids['lng'] = df_centroids['lng'].astype(float)

    for i in clusters:
        points = df_points[df_points['cluster'] == str(i)].loc[:, ['lat', 'lng']].to_numpy()
        c = centroid(MultiPoint(points))
        df_centroids.loc[i:i, 'lat'] = c.x
        df_centroids.loc[i:i, 'lng'] = c.y

    return calcul_distances(df_centroids)

def draw_centroids(map_index: int, df_centers: pd.DataFrame):
    st.session_state[f"markers_center_{map_index}"] = []

    for index, row in df_centers.iterrows():
        st.session_state[f"markers_center_{map_index}"].append(
            folium.CircleMarker(
                location=[
                    row['lat'],
                    row['lng'],
                ],
                tooltip=str(index),
                fill=False,
                color=COLORS[int(index)],
                weight=5,
                opacity=1,
                radius=10,
            )
        )

def draw_scores(df_points: pd.DataFrame):
    measures = {
        'kmeans': [],
        'gaussian': []
    }

    n_clusters_list = range(
        1,
        min(
            df_points.shape[0],
            len(COLORS) - 1
        ) + 1
    )

    points = df_points.loc[:,['x', 'y']].to_numpy()

    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k)
        gaussian = GaussianMixture(n_components=k, random_state=42)

        kmeans.fit(points)
        gaussian.fit(points)

        measures['kmeans'].append(kmeans.inertia_)
        measures['gaussian'].append(gaussian.bic(points))

    measures['kmeans'] = StandardScaler().fit_transform(np.array(measures['kmeans']).reshape(-1, 1)).reshape(-1)
    measures['gaussian'] = StandardScaler().fit_transform(np.array(measures['gaussian']).reshape(-1, 1)).reshape(-1)

    fig = px.line(
        pd.DataFrame(measures, index=n_clusters_list),
        title='Méthode du Coude',
        labels={
            'index': 'Nombre de Clusters',
            'value': "Score",
        },
    )
    fig.update_layout(showlegend=True)

    return fig

def map_prepare(map_index: int | None = None):
    if map_index is None:
        map_index = 0

    if f"center_{map_index}" not in st.session_state:
        st.session_state[f"center_{map_index}"] = CENTER_START
    if f"zoom_{map_index}" not in st.session_state:
        st.session_state[f"zoom_{map_index}"] = ZOOM_START

    if f"markers_point_{map_index}" not in st.session_state:
        st.session_state[f"markers_point_{map_index}"] = []
    if f"markers_center_{map_index}" not in st.session_state:
        st.session_state[f"markers_center_{map_index}"] = []
    if f"markers_line_{map_index}" not in st.session_state:
        st.session_state[f"markers_line_{map_index}"] = []

    if f"clustering_algo_{map_index}" not in st.session_state:
        st.session_state[f"clustering_algo_{map_index}"] = None
    if f"n_clusters_kmeans_{map_index}" not in st.session_state:
        st.session_state[f"n_clusters_kmeans_{map_index}"] = None
    if f"n_clusters_gaussian_{map_index}" not in st.session_state:
        st.session_state[f"n_clusters_gaussian_{map_index}"] = None

    if f"df_points_{map_index}" not in st.session_state:
        st.session_state[f"df_points_{map_index}"] = None
    if f"df_centroids_{map_index}" not in st.session_state:
        st.session_state[f"df_centroids_{map_index}"] = None

def map_clear(map_index: int):
    map_start(map_index)
    st.session_state[f"markers_point_{map_index}"] = []
    st.session_state[f"markers_center_{map_index}"] = []
    st.session_state[f"markers_line_{map_index}"] = []

    st.session_state[f"clustering_algo_{map_index}"] = None
    st.session_state[f"n_clusters_kmeans_{map_index}"] = None
    st.session_state[f"n_clusters_gaussian_{map_index}"] = None

    st.session_state[f"df_points_{map_index}"] = None
    st.session_state[f"df_centroids_{map_index}"] = None


def map_start(map_index: int):
    st.session_state[f"zoom_{map_index}"] = ZOOM_START + random.uniform(0.00001, 0.00009)
    st.session_state[f"center_{map_index}"] = [
        CENTER_START[0] + random.uniform(0.00001, 0.00009),
        CENTER_START[1] + random.uniform(0.00001, 0.00009),
    ]

def map_draw(map_index: int = 0):
    map_prepare(map_index)

    m = folium.Map(
        location=CENTER_START,
        zoom_start=ZOOM_START,
        min_zoom=ZOOM_START,
        max_bounds=True,
        min_lat=MIN_LAT,
        max_lat=MAX_LAT,
        min_lon=MIN_LNG,
        max_lon=MAX_LNG,
    )

    fg = folium.FeatureGroup(name="Markers")

    folium.PolyLine(
        locations=[
            [MIN_LAT, MIN_LNG],
            [MAX_LAT, MIN_LNG],
            [MAX_LAT, MAX_LNG],
            [MIN_LAT, MAX_LNG],
            [MIN_LAT, MIN_LNG],
        ],
        color="#FF0000",
        weight=2,
    ).add_to(fg)

    for marker in st.session_state[f"markers_point_{map_index}"]:
        fg.add_child(marker)

    for marker in st.session_state[f"markers_center_{map_index}"]:
        fg.add_child(marker)

    for marker in st.session_state[f"markers_line_{map_index}"]:
        fg.add_child(marker)

    st_folium(
        m,
        center=st.session_state[f"center_{map_index}"],
        zoom=st.session_state[f"zoom_{map_index}"],
        width='100%',
        feature_group_to_add=fg,
        returned_objects=['center', 'zoom'],
        key=f"map_{map_index}",
    )

def prepare_clustering(map_index: int, el: DeltaGenerator, type: str):
    n_clusters_list = range(
        1,
        min(
            st.session_state[f"df_points_{map_index}"].shape[0],
            len(COLORS) - 1
        ) + 1
    )

    st.session_state[f"n_clusters_{type}_{map_index}"] = el.slider(
        "Nombre de clusters :",
        min_value=n_clusters_list[0],
        max_value=n_clusters_list[-1],
        step=1,
        label_visibility='collapsed',
        key=f'k_{type}_{map_index}',
    )

    return st.session_state[f"df_points_{map_index}"].loc[:,['x', 'y']].to_numpy()

def finish_clustering(map_index: int, el: DeltaGenerator, type: str):
    st.session_state[f"df_points_{map_index}"][f'cluster_{type}'] = \
        st.session_state[f"df_points_{map_index}"][f'cluster_{type}'].astype(str)

    fig = px.scatter(
        st.session_state[f"df_points_{map_index}"].reset_index(),
        x='x',
        y='y',
        color=f'cluster_{type}',
        hover_name='index',
        height=700,
        size=[1] * st.session_state[f"df_points_{map_index}"].shape[0],
        color_discrete_sequence=COLORS,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    el.plotly_chart(fig, use_container_width=True, key=f'fig_{type}_{map_index}')

def draw_kmeans(map_index: int, el: DeltaGenerator):
    el.text('K-means')

    points = prepare_clustering(
        map_index,
        el,
        'kmeans',
    )

    model = KMeans(n_clusters=st.session_state[f"n_clusters_kmeans_{map_index}"])
    model.fit(points)
    st.session_state[f"df_points_{map_index}"]['cluster_kmeans'] = model.predict(points)

    finish_clustering(
        map_index,
        el,
        'kmeans',
    )

def draw_gaussian(map_index: int, el: DeltaGenerator):
    el.text('Gaussian Mixture')

    points = prepare_clustering(
        map_index,
        el,
        'gaussian',
    )

    model = GaussianMixture(n_components=st.session_state[f"n_clusters_gaussian_{map_index}"], random_state=42)
    model.fit(points)
    st.session_state[f"df_points_{map_index}"]['cluster_gaussian'] = model.predict(points)

    finish_clustering(
        map_index,
        el,
        'gaussian',
    )
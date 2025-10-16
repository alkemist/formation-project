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

def random_cities(df_cities_random, n: int):
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

def calcul_distances(df: pd.DataFrame, reset_clusters = True):
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

        df = pd.merge(
            df,
            df_points,
            left_index=True,
            right_index=True,
        )

    return df.sort_index().reset_index(), df_distances


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

    df_points = df_points.sort_values(by='order')
    df_points['index'] = df_points.index

    df_points['next'] = df_points['index'].shift(periods=-1, fill_value='')
    df_points['next'] = df_points['next'].astype(str)
    df_points.loc[df_points.index[-1], 'next'] = df_points.iloc[0]['index']

    df_points['prev'] = df_points['index'].shift(periods=1, fill_value='')
    df_points['prev'] = df_points['prev'].astype(str)
    df_points.loc[df_points.index[0], 'prev'] = df_points.iloc[-1]['index']

    df_points["distance"] = df_points.apply(
        lambda r: df_distances.loc[r["index"], r["next"]] if r["next"] else 0,
        axis=1
    )
    #df_points["distance_cum"] = df_points["distance"].cumsum()

    return df_points.drop(columns=['index'])

def calcul_centroids(df_cluster_points: pd.DataFrame):
    clusters = df_cluster_points['cluster'].unique()

    df_centroids = pd.DataFrame()

    for i in clusters:
        cluster_points = df_cluster_points[df_cluster_points['cluster'] == str(i)].loc[:, ['lat', 'lng']].to_numpy()
        c = centroid(MultiPoint(cluster_points))

        df_centroids = pd.concat([
            df_centroids,
            pd.DataFrame({
                'index': [i],
                'lat': [c.x],
                'lng': [c.y]
            })
        ], ignore_index=True)

    df_centroids['index'] = df_centroids['index'].astype(str)
    df_centroids['lat'] = df_centroids['lat'].astype(float)
    df_centroids['lng'] = df_centroids['lng'].astype(float)

    return calcul_distances(df_centroids.set_index('index'))

    return calcul_distances(df_centroids.set_index('index'))
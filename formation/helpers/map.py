import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sqlalchemy import select, or_, and_
import streamlit as st

from formation import engine
from formation.data.constants import BEST_PATH_MAX_COMBINATIONS
from formation.models import Point, Layer
from formation.models.distance import Distance
from itertools import combinations, permutations, product


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

def calcul_distances(df: pd.DataFrame):
    indexes = df.index.unique()

    indexes_combinations = list(combinations(indexes, 2))

    df_distances_pairs = pd.DataFrame()

    for (point_a, point_b) in indexes_combinations:
        df_distances_pairs = pd.concat([
            df_distances_pairs,
                pd.DataFrame({
                    'code_in': [point_a],
                    'code_out': [point_b],
                    'lat_in': [df.loc[point_a:point_a, 'lat'].values[0]],
                    'lng_in': [df.loc[point_a:point_a, 'lng'].values[0]],
                    'lat_out': [df.loc[point_b:point_b, 'lat'].values[0]],
                    'lng_out': [df.loc[point_b:point_b, 'lng'].values[0]],
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

    df = set_xy(df, df_distances_pairs)

    return df.sort_index().reset_index(), df_distances_pairs

def set_xy(df, df_distances_pairs):
    df_pairs_symetry = pd.concat(
        [
            df_distances_pairs,
            df_distances_pairs.rename(columns={
                'code_in': 'code_out',
                'code_out': 'code_in',
                'lat_in': 'lat_out',
                'lat_out': 'lat_in',
                'lng_in': 'lng_out',
                'lng_out': 'lng_in',
            })
        ],
        ignore_index=True,
    )

    df_distances = df_pairs_symetry[['code_in', 'code_out', 'distance']].rename(columns={'code_in': 'code'}) \
        .pivot(index='code', columns='code_out', values='distance') \
        .fillna(0)

    if df_distances.shape[0] > 1:
        model = MDS(n_components=2, dissimilarity='precomputed', random_state=1, n_init=1)
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

    return df

def get_distances(layer: Layer) -> pd.DataFrame:
    point_in_exists = select(Point).where(
        (Point.layer_id == layer.id) &
        (Point.lat == Distance.lat_in) &
        (Point.lng == Distance.lng_in)
    ).exists()

    point_out_exists = select(Point).where(
        (Point.layer_id == layer.id) &
        (Point.lat == Distance.lat_out) &
        (Point.lng == Distance.lng_out)
    ).exists()

    st_distances = select(Distance).where(
        and_(
            point_in_exists,
            point_out_exists
        )
    )

    df_pairs = pd.read_sql(st_distances, engine)

    df_pairs['lat_in'] = df_pairs['lat_in'].astype(str)
    df_pairs['lng_in'] = df_pairs['lng_in'].astype(str)
    df_pairs['lat_out'] = df_pairs['lat_out'].astype(str)
    df_pairs['lng_out'] = df_pairs['lng_out'].astype(str)

    df_pairs['code_in'] = df_pairs[['lat_in', 'lng_in']].agg(':'.join, axis=1)
    df_pairs['code_out'] = df_pairs[['lat_out', 'lng_out']].agg(':'.join, axis=1)

    df_pairs_symetry = pd.concat(
        [
            df_pairs,
            df_pairs.rename(columns={
                'code_in': 'code_out',
                'code_out': 'code_in',
            })
        ],
        ignore_index=True,
    )\
        [['code_in', 'code_out', 'distance']]\
        .drop_duplicates(subset=['code_in', 'code_out'])

    return df_pairs_symetry.rename(columns={'code_in': 'code'}) \
        .pivot(index='code', columns='code_out', values='distance') \
        .fillna(0)

def calc_next(df_points: pd.DataFrame, df_points_distances: pd.DataFrame) -> pd.DataFrame:
    df_points['lat'] = df_points['lat'].astype(str)
    df_points['lng'] = df_points['lng'].astype(str)
    df_points['latlng'] = df_points[['lat', 'lng']].agg(':'.join, axis=1)

    df_points['code_next'] = df_points['code'].shift(periods=-1, fill_value='')
    df_points['code_next'] = df_points['code_next'].astype(str)
    df_points.loc[df_points.index[-1], 'code_next'] = df_points.iloc[0]['code']

    df_points['latlng_next'] = df_points['latlng'].shift(periods=-1, fill_value='')
    df_points['latlng_next'] = df_points['latlng_next'].astype(str)
    df_points.loc[df_points.index[-1], 'latlng_next'] = df_points.iloc[0]['latlng']

    df_points['lat_next'] = df_points['lat'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'lat_next'] = df_points.iloc[0]['lat']

    df_points['lng_next'] = df_points['lng'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'lng_next'] = df_points.iloc[0]['lng']

    df_points['distance'] = df_points.apply(lambda x: df_points_distances[x['latlng']][x['latlng_next']], axis=1)

    return df_points

def best_path(df_centroids: pd.DataFrame, df_points: pd.DataFrame, df_points_distances: pd.DataFrame) -> pd.DataFrame | None:
    df_points['lat'] = df_points['lat'].astype(str)
    df_points['lng'] = df_points['lng'].astype(str)
    df_points['latlng'] = df_points[['lat', 'lng']].agg(':'.join, axis=1)
    
    cluster_indexes = list(df_centroids['code'].unique())
    cluster_count = len(cluster_indexes)

    with st.spinner(f"- [Etape 1] avec {cluster_count} clusters", show_time=True):
        tours_indexes = [
            (
                cluster_indexes[i],
                cluster_indexes[(i + 1) % cluster_count],
                cluster_indexes[(i + 2) % cluster_count]
            )
            for i in range(cluster_count)
        ]

        tours_dict = {}

        for (cluster_prev, cluster_current, cluster_next) in tours_indexes:
            # print("-- Step :", (cluster_prev, cluster_current, cluster_next))

            bridge = (cluster_prev, cluster_current, cluster_next)

            tours_dict[bridge] = []

            df_cluster_current = df_points[df_points['cluster'] == cluster_current]
            df_cluster_prev = df_points[df_points['cluster'] == cluster_prev]
            df_cluster_next = df_points[df_points['cluster'] == cluster_next]

            bridges_prev = list(
                product(
                    df_cluster_prev['latlng'].unique(),
                    df_cluster_current['latlng'].unique(),
                )
            )

            bridges_next = list(
                product(
                    df_cluster_current['latlng'].unique(),
                    df_cluster_next['latlng'].unique(),
                )
            )

            bridge_paths = list(
                product(bridges_prev, bridges_next)
            )

            for ((cluster_prev_in, cluster_current_in), (cluster_current_out, cluster_next_out)) in bridge_paths:
                if cluster_current_in != cluster_current_out or df_cluster_current.shape[0] == 1:
                    if df_cluster_current.shape[0] == 1 and cluster_current_in == cluster_current_out:
                        cluster_path = [cluster_current_in]
                        if cluster_path not in tours_dict[bridge]:
                            # print("--- Path :", cluster_path)
                            tours_dict[bridge].append(cluster_path)
                    else:
                        cluster_between = df_cluster_current[
                            (df_cluster_current['latlng'] != cluster_current_in) &
                            (df_cluster_current['latlng'] != cluster_current_out)
                            ]['latlng'].values

                        for between in list(permutations(cluster_between)):
                            cluster_path = [cluster_current_in] + list(between) + [cluster_current_out]

                            if cluster_path not in tours_dict[bridge]:
                                # print("--- Path :", cluster_path)
                                tours_dict[bridge].append(cluster_path)

        values = list(tours_dict.values())
        paths_pairs = []
        paths = []

        combinations_list = list(product(*values))

    with st.spinner(f"- [Etape 2] avec {len(combinations_list)} combinaisons", show_time=True):
        if len(combinations_list) > BEST_PATH_MAX_COMBINATIONS:
            return None

        for combination in combinations_list:
            path = []
            for step in combination:
                path = path + step

            pairs = list(zip(path[:-1], path[1:]))
            pairs.append((path[-1], path[0]))

            paths.append(path)
            paths_pairs.append(pairs)

        distances = dict()
        for i, path in enumerate(paths_pairs):
            distance = 0
            for point_a, point_b in path:
                distance += df_points_distances[point_a][point_b]

            distances[tuple(path)] = distance

        tours_best_distance = min(distances.values())
        tours_best_pairs = min(distances, key=distances.get)

    points_prev, points_next = zip(*tours_best_pairs)

    return pd.DataFrame(
        index=list(points_prev)
    ).merge(
        df_points,
        left_index=True,
        right_on='latlng',
        how='left',
    ).reset_index(drop=True)

def sort_points_by_cluster(df_centroids: pd.DataFrame, df_points: pd.DataFrame) -> pd.DataFrame:
    centroids_order = df_centroids['code'].to_dict()
    centroids_order = {v: k for k, v in centroids_order.items()}

    df_points['order'] = df_points['cluster'].apply(lambda x: centroids_order[x])
    df_points = df_points.sort_values('order').reset_index(drop=True).drop(columns=['order'])

    return df_points

def calcul_min_local_path(df_points: pd.DataFrame, df_points_distances: pd.DataFrame, max_rollback: int) -> pd.DataFrame:
    n = 0

    df_points = calc_next(df_points, df_points_distances)
    distance_total_prev = float(df_points['distance'].sum())

    while True:
        df_points, distance_total = min_local_path(df_points, df_points_distances, max_rollback, n)

        gap = distance_total_prev - distance_total
        distance_total_prev = distance_total

        if gap < 10:
            break

        n += 1

    return df_points

def min_local_path(df_points: pd.DataFrame, df_points_distances: pd.DataFrame, max_rollback: int, n) -> (pd.DataFrame, float):
    distance_total = float(df_points['distance'].sum())
    indexes_prev = (df_points.index[-max_rollback:] if max_rollback < df_points.shape[0] else df_points.index).tolist()[::-1]

    with st.spinner(f'- [Etape 4.{n}] avec {df_points.shape[0]} points et une distance min de {distance_total}', show_time=True):
        for index_current, point in df_points.iterrows():
            # print(f"-- Index {point['code']} with prev {indexes_prev}")
            local_switches = {}

            for index_prev in indexes_prev:
                df_points_copy = df_points.copy()

                df_points_copy.iloc[[index_current, index_prev]] = df_points_copy.iloc[[index_prev, index_current]].values
                df_points_copy = calc_next(df_points_copy, df_points_distances)
                distance_total_new = df_points_copy['distance'].sum()

                if distance_total_new < distance_total:
                    local_switches[(index_current, index_prev)] = float(distance_total_new)

            indexes_prev = [index_current] + indexes_prev

            if len(local_switches.items()) > 0:
                (index_current, index_prev) = min(local_switches, key=local_switches.get)
                # print(f"-- Best : ", local_switches)
                # print(f"-- Inverses : ", index_current, 'with', index_prev)

                df_points.iloc[[index_current, index_prev]] = df_points.iloc[[index_prev, index_current]].values
                df_points = calc_next(df_points, df_points_distances)
                distance_total = float(df_points['distance'].sum())

                i_index_current = indexes_prev.index(index_current)
                i_index_prev = indexes_prev.index(index_prev)
                indexes_prev[i_index_current], indexes_prev[i_index_prev] = \
                    indexes_prev[i_index_prev], indexes_prev[i_index_current]
                index_current, index_prev = index_prev, index_current

            indexes_prev = indexes_prev[1:]

            # print(f"-- New prev :", indexes_prev)

            for i, index_prev in enumerate(indexes_prev):
                index_prev_prev = indexes_prev[i]
                indexes_prev[i] = index_current
                index_current = index_prev_prev

    return df_points, distance_total
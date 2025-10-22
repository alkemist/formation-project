import os
from datetime import datetime
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
from itertools import permutations, product

from formation.helpers.old.map import calc_next, draw_map, calcul_centroids

COLORS = px.colors.qualitative.Vivid + px.colors.qualitative.Alphabet

import plotly.io as pio

pio.get_chrome()

def best_path(df_centroids: pd.DataFrame, df_points: pd.DataFrame, df_points_distances: pd.DataFrame) -> pd.DataFrame | None:
    cluster_indexes = list(df_centroids['index'].unique())
    cluster_count = len(cluster_indexes)

    print(f"- [Step 1] with {cluster_count} clusters")
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
                df_cluster_prev['index'].unique(),
                df_cluster_current['index'].unique(),
            )
        )

        bridges_next = list(
            product(
                df_cluster_current['index'].unique(),
                df_cluster_next['index'].unique(),
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
                        (df_cluster_current['index'] != cluster_current_in) &
                        (df_cluster_current['index'] != cluster_current_out)
                        ]['index'].values

                    for between in list(permutations(cluster_between)):
                        cluster_path = [cluster_current_in] + list(between) + [cluster_current_out]

                        if cluster_path not in tours_dict[bridge]:
                            # print("--- Path :", cluster_path)
                            tours_dict[bridge].append(cluster_path)

    values = list(tours_dict.values())
    paths_pairs = []
    paths = []

    combinations_list = list(product(*values))

    print(f"- [Step 2] with {len(combinations_list)} combinaisons")

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

    print(f"- [Step 3] best path with distance min {tours_best_distance}")

    points_prev, points_next = zip(*tours_best_pairs)

    return pd.DataFrame({
        'point_prev': list(points_prev),
        'point_next': list(points_next),
    }).merge(
        df_points,
        left_on='point_prev',
        right_on='index',
        how='left',
    ).merge(
        df_points,
        left_on='point_next',
        right_on='index',
        how='left',
        suffixes=('', '_next'),
    ).drop(columns=['point_prev', 'point_next'])

def sort_points_by_cluster(df_centroids: pd.DataFrame, df_points: pd.DataFrame) -> pd.DataFrame:
    centroids_order = df_centroids['index'].to_dict()
    centroids_order = {v: k for k, v in centroids_order.items()}

    df_points['order'] = df_points['cluster'].apply(lambda x: centroids_order[x])
    df_points = df_points.sort_values('order').reset_index(drop=True).drop(columns=['order'])

    return df_points

def calcul_min_local_path(df_points: pd.DataFrame, df_points_distances: pd.DataFrame) -> pd.DataFrame:
    n = 0

    df_points = calc_next(df_points, df_points_distances)
    distance_total_prev = float(df_points['distance'].sum())

    while True:
        df_points, distance_total = min_local_path(df_points, df_points_distances, n)

        gap = distance_total_prev - distance_total
        distance_total_prev = distance_total

        if gap < 10:
            break

        n += 1

    return df_points

def min_local_path(df_points: pd.DataFrame, df_points_distances: pd.DataFrame, n) -> (pd.DataFrame, float):
    distance_total = float(df_points['distance'].sum())
    indexes_prev = df_points.index[-MAX_ROLLBACK_CALCUL_MIN_PATH:].tolist()[::-1]

    print(f'- [Step 4.{n}] with {df_points.shape[0]} points and distance min {distance_total}')

    for index_current, point in df_points.iterrows():
        # print(f"-- Index {point['index']} with prev {indexes_prev}")
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

print(f"-- Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

CLUSTER_MIN_LENGTH = 3
CLUSTER_MAX_LENGTH = 3
BEST_PATH_MAX_COMBINATIONS = 80000
BEST_PATH_MAX_CLUSTERS = 20
MAX_ROLLBACK_CALCUL_MIN_PATH = 100

INPUT_DIR = 'data/formated/layers/'
OUTPUT_DIR = 'data/formated/layers/method_2/'
GRAPH_DIR = 'export/graphs/'
MAP_DIR = 'export/map/'

df_points_coords = pd.read_csv(INPUT_DIR + 'layer_1_points.csv', dtype={'index': str})
df_points_distances = pd.read_csv(INPUT_DIR + 'layer_1_distances.csv', dtype={'index': str}, index_col=0)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)
if not os.path.exists(MAP_DIR):
    os.makedirs(MAP_DIR)

layers_count = 1

while True:
    print(f'Layer {layers_count} with {df_points_coords.shape[0]} points')

    points = df_points_coords.loc[:,['x', 'y']].to_numpy()

    k_kmeans = 1
    points_by_cluster = None

    if df_points_coords.shape[0] > CLUSTER_MIN_LENGTH:
        if CLUSTER_MAX_LENGTH ** 2 > df_points_coords.shape[0]:
            k_kmeans = CLUSTER_MIN_LENGTH
        else:
            k_kmeans = df_points_coords.shape[0] // CLUSTER_MAX_LENGTH

    print(f'- with {k_kmeans} clusters with {points_by_cluster} max size')

    while points_by_cluster is None or points_by_cluster > CLUSTER_MAX_LENGTH:
        model_kmeans = KMeans(n_clusters=k_kmeans)
        model_kmeans.fit(points)
        df_points_coords['cluster'] = model_kmeans.predict(points)
        df_points_coords['cluster'] = df_points_coords['cluster'].astype(str)

        points_by_cluster = df_points_coords[['cluster', 'index']].groupby('cluster').count().max().values[0]

        if points_by_cluster > CLUSTER_MAX_LENGTH:
            k_kmeans += 1

    print(f'- with {k_kmeans} clusters with {points_by_cluster} max size')

    draw_map(df_points_coords.reset_index()).save(f'{MAP_DIR}layer_{layers_count}_map.html')

    df_points_coords.to_csv(f'{OUTPUT_DIR}layer_{layers_count}_points.csv', index=False)
    df_points_distances.to_csv(f'{OUTPUT_DIR}layer_{layers_count}_distances.csv')

    fig = px.scatter(
        df_points_coords.reset_index(),
        x='x',
        y='y',
        color='cluster',
        hover_name='index',
        height=700,
        size=[1] * df_points_coords.shape[0],
    )
    fig.write_image(f"{GRAPH_DIR}layer_{layers_count}_points.jpg")

    if df_points_coords.shape[0] > CLUSTER_MIN_LENGTH:
        df_centroids_coords, df_centroids_distances = calcul_centroids(df_points_coords)

        df_points_coords, df_points_distances = df_centroids_coords, df_centroids_distances

        layers_count += 1
    else:
        break

layers_list = range(2, layers_count + 1)[::-1]
print("steps :", list(layers_list))
df_tour = pd.DataFrame()

n = layers_list[0]
df_centroids = pd.read_csv(OUTPUT_DIR + 'layer_' + str(n) + '_points.csv', dtype={'index': str, 'cluster': str})

for n in layers_list:
    df_cluster = None

    df_centroids_distances = pd.read_csv(OUTPUT_DIR + 'layer_' + str(n) + '_distances.csv', dtype={'index': str}, index_col=0)

    print(f'Begin layer {n - 1} with {df_centroids.shape[0]} points')

    df_points = pd.read_csv(OUTPUT_DIR + 'layer_' + str(n - 1) + '_points.csv', dtype={'index': str, 'cluster': str})

    df_points_distances = pd.read_csv(OUTPUT_DIR + 'layer_' + str(n - 1) + '_distances.csv', dtype={'index': str}, index_col=0)

    if df_centroids.shape[0] < BEST_PATH_MAX_CLUSTERS:
        # Le meilleur chemin est calculé par rapport aux groupes précédents
        df_cluster = best_path(df_centroids, df_points, df_points_distances)

    if df_cluster is None:
        df_cluster = sort_points_by_cluster(df_centroids, df_points)

    # Le min local est calculé indépedemment des groupes
    df_cluster = calcul_min_local_path(df_cluster, df_points_distances)

    df_cluster.to_csv(OUTPUT_DIR + 'layer_' + str(n - 1) + '_cluster.csv', index=False)
    draw_map(df_cluster, True).save(f'{MAP_DIR}layer_{n - 1}_map.html')
    df_centroids = df_cluster[['index', 'lat', 'lng', 'x', 'y', 'cluster']]

    print(f'End layer {n - 1} with {df_centroids.shape[0]} points')

print(f"-- End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
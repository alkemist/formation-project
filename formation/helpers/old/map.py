from folium import CircleMarker, PolyLine, Map, FeatureGroup

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from shapely import MultiPoint, centroid



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

    model = MDS(n_components=2, dissimilarity='precomputed', random_state=1, n_init=1)
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

def calc_next(df_points: pd.DataFrame, df_points_distances: pd.DataFrame) -> pd.DataFrame:
    df_points['index_next'] = df_points['index'].shift(periods=-1, fill_value='')
    df_points['index_next'] = df_points['index_next'].astype(str)
    df_points.loc[df_points.index[-1], 'index_next'] = df_points.iloc[0]['index']

    df_points['cluster_next'] = df_points['cluster'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'cluster_next'] = df_points.iloc[0]['cluster']

    df_points['lat_next'] = df_points['lat'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'lat_next'] = df_points.iloc[0]['lat']

    df_points['lng_next'] = df_points['lng'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'lng_next'] = df_points.iloc[0]['lng']

    df_points['x_next'] = df_points['x'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'x_next'] = df_points.iloc[0]['x']

    df_points['y_neyt'] = df_points['y'].shift(periods=-1, fill_value='')
    df_points.loc[df_points.index[-1], 'y_neyt'] = df_points.iloc[0]['y']

    df_points['distance'] = df_points.apply(lambda x: df_points_distances[x['index']][x['index_next']], axis=1)

    return df_points

def draw_map(df_points, end=False):
    m = Map(
        location=(46.227638, 2.213749),
        zoom_start=6,
    )

    fg = FeatureGroup(name="Markers")

    for i, point in df_points.iterrows():
        if 'lat' in point and 'lng' in point:
            fg.add_child(
                CircleMarker(
                    location=[
                        point['lat'],
                        point['lng'],
                    ],
                    tooltip=f"{point['index']} ({point['cluster']})",
                    fill=True,
                    #color=COLORS[int(point['cluster'])],
                    weight=0,
                    fill_opacity=0.6,
                    radius=15,
                )
            )

        if 'center_lat' in point and 'center_lng' in point:
            fg.add_child(
                CircleMarker(
                    location=[
                        point['center_lat'],
                        point['center_lng'],
                    ],
                    #tooltip=f"{point['index']} ({point['cluster_current']})",
                    fill=True,
                    #color=COLORS[int(point['cluster_current'])],
                    weight=0,
                    fill_opacity=0.6,
                    radius=15,
                )
            )

        if 'lat_next' in point and 'lng_next' in point:
            fg.add_child(
                PolyLine(
                    locations=[
                        [point['lat'], point['lng']],
                        [point['lat_next'], point['lng_next']],
                    ],
                    color="black",
                    weight=4,
                )
            )

    m.add_child(fg)

    return m
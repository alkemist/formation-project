import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from shapely import MultiPoint, centroid
from sklearn import cluster

def precalcul_clusters(df_points: pd.DataFrame, size: int) :
    points = df_points.loc[:,['x', 'y']].to_numpy()

    k_first = 1

    if df_points.shape[0] > size:
        if size ** 2 > df_points.shape[0]:
            k_first = size
        else:
            k_first = df_points.shape[0] // size

    return points, k_first

def postcalcul_clusters(df_points: pd.DataFrame, size: int, k: int):
    df_points['cluster'] = df_points['cluster'].astype(str)

    points_by_cluster = df_points[['cluster', 'code']].groupby('cluster').count().max().values[0]

    if points_by_cluster > size:
        k += 1

    return df_points, points_by_cluster, k

def calcul_clusters_kmeans(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.KMeans(n_clusters=k, random_state=42, **params)
        model.fit(points)
        df_points['cluster'] = model.predict(points)

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_gaussian(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = GaussianMixture(n_components=k, random_state=42, **params)
        model.fit(points)
        df_points['cluster'] = model.predict(points)

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_dbscan(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.DBSCAN(min_samples=k, **params)
        model.fit(points)
        df_points['cluster'] = model.labels_

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_affinity(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.AffinityPropagation(random_state=42, **params)
        model.fit(points)
        df_points['cluster'] = model.predict(points)

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_meanshift(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.MeanShift(bandwidth=42, **params)
        model.fit(points)
        df_points['cluster'] = model.predict(points)

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_spectral(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.SpectralClustering(n_clusters=k, **params)
        model.fit(points)
        df_points['cluster'] = model.labels_

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_agglomerative(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.AgglomerativeClustering(n_clusters=k, **params)
        model.fit(points)
        df_points['cluster'] = model.labels_

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_hdbscan(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.HDBSCAN(min_cluster_size=k, **params)
        model.fit(points)
        df_points['cluster'] = model.labels_

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_optics(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.OPTICS(min_samples=k, **params)
        model.fit(points)
        df_points['cluster'] = model.labels_

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_clusters_birch(df_points: pd.DataFrame, size: int, params: dict) :
    points, k = precalcul_clusters(df_points, size)
    points_by_cluster = None

    while (points_by_cluster is None or points_by_cluster > size) and k < df_points.shape[0]:
        model = cluster.Birch(n_clusters=k, **params)
        model.fit(points)
        df_points['cluster'] = model.predict(points)

        df_points, points_by_cluster, k = postcalcul_clusters(df_points, size, k)

    return df_points

def calcul_centroids(df_cluster_points: pd.DataFrame) -> pd.DataFrame:
    clusters = df_cluster_points['cluster'].unique()

    df_centroids = pd.DataFrame()

    for i in clusters:
        cluster_points = df_cluster_points[df_cluster_points['cluster'] == str(i)].loc[:, ['lat', 'lng']].to_numpy()
        c = centroid(MultiPoint(cluster_points))

        df_centroids = pd.concat([
            df_centroids,
            pd.DataFrame({
                'code': [i],
                'lat': [c.x],
                'lng': [c.y]
            })
        ], ignore_index=True)

    df_centroids['code'] = df_centroids['code'].astype(str)
    df_centroids['lat'] = df_centroids['lat'].astype(float)
    df_centroids['lng'] = df_centroids['lng'].astype(float)

    return df_centroids

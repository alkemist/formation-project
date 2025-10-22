import numpy as np
import pandas as pd
from sklearn.manifold import MDS

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

    df_distances = df_pairs.rename(columns={'point_a': 'code'}) \
        .pivot(index='code', columns='point_b', values='distance') \
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

    return df.sort_index().reset_index(), df_distances


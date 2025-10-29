import itertools

import streamlit as st
import pandas as pd

from formation.data.constants import BEST_PATH_MAX_CLUSTERS
from formation.helpers import clustering
from formation.helpers.layer import save_layer
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from formation import engine, session
from formation.data.pages import list_pages
from formation.helpers.map import calcul_distances, get_distances, best_path, sort_points_by_cluster, \
    calcul_min_local_path
from formation.helpers.streamlit import goto_page_object
from formation.models import Batch, Point, Layer, Configuration
from formation.models.distance import Distance

st.title(list_pages['batches']['title'])
st.set_page_config(
    page_title=list_pages['batches']['title'],
    page_icon=list_pages['batches']['icon'],
    layout="wide",
)

INPUT_DIR = 'data/formated/'

tab_list, tab_create = st.tabs(["Liste des lots", "Créer un lot"])

with tab_list:
    st_batches = select(Batch, Layer)\
        .filter(Batch.id == Layer.batch_id)\
        .order_by(Batch.id)
    df_batches = pd.read_sql(st_batches, engine) \
        .groupby(['id']) \
        .agg(
            count_configurations=('id_1', 'nunique'),
            count_points=('points_count', 'first'),
    ) \
        .reset_index()

    df_batches['link'] = '/batch?batch_id=' + df_batches['id'].astype(str)

    st.dataframe(
        df_batches,
        column_config={
            "id": st.column_config.Column(
                'Id',
                pinned=True,
            ),
            "count_configurations": st.column_config.NumberColumn(
                'Configurations',
            ),
            "count_points": st.column_config.NumberColumn(
                'Points',
            ),
            "link": st.column_config.LinkColumn(
                "Liste",
                max_chars=100,
                display_text=":material/visibility:",
                pinned=False,
                width=100,
            ),
        },
        hide_index=True,
    )

batch = None

MAX_ROLLBACK_CALCUL_MIN_PATH = [10, 30]
CLUSTERING_LENGTHS = [3, 4]
CLUSTERING_CONFIGURATIONS = {
    'kmeans': {},
    'gaussian': {
        'covariance_type': ['full', 'tied', 'diag', 'spherical']
    },
    # 'dbscan': {
    #     # 'eps': [0.5, 3, 5, 10]
    # },
    # 'hdbscan': {
    #     # 'cluster_selection_epsilon': [0.5, 3, 5, 10],
    #     # 'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'],
    #     # 'algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree'],
    #     # 'cluster_selection_method': ['eom', 'leaf'],
    # },
    # 'affinity': {
    #     # 'damping': [0.5, 1],
    #     # 'affinity': ['euclidean', 'precomputed'],
    # },
    # 'meanshift': {},
    # 'spectral': {}, # TROP DE MEMOIRE
    'agglomerative': {
        'linkage': ['ward', 'complete', 'average', 'single'],
        # 'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'],
    },
    # 'optics': {
    #     # 'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'cityblock'],
    #     # 'cluster_method': ['xi', 'dbscan'],
    #     # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    # },
    'birch': {
        'compute_labels': [True, False],
    },
}

with tab_create:
    top_left, top_middle, top_right = st.columns([2, 1, 2])

    top_left.text("Créer un lot avec")

    city_count = st.session_state["city_count"] if "city_count" in st.session_state else 3

    st.session_state["city_count"] = top_middle.number_input("", value=city_count, min_value=3, max_value=255, step=1, label_visibility='collapsed', width='stretch')

    if top_right.button("Selectionner les villes", type="primary", use_container_width=True):
        with st.spinner("Selection des villes", show_time=True):
            df_cities = pd.read_csv(INPUT_DIR + 'cities-france.csv',
                                    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
                                    ).rename(columns={'code_insee': 'code', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'}) \
                .set_index('code') \
                .loc[:, ['lat', 'lng']]
            df_cities['cluster'] = None

            st.session_state["points"], st.session_state["distances"] \
                = calcul_distances(df_cities.sample(st.session_state["city_count"]))

    if "points" in st.session_state and st.session_state["points"] is not None and \
            "distances" in st.session_state and st.session_state["distances"] is not None:

        st.dataframe(
            st.session_state["points"].drop(columns=['cluster']),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Créer le lot", type="primary", use_container_width=True):
            df_points = st.session_state["points"]
            df_distances_pairs = st.session_state["distances"]
            del st.session_state["points"]
            del st.session_state["distances"]

            with engine.connect() as conn:
                try:
                    with st.spinner(f"Calcul des configurations", show_time=True):
                        batch_id = conn.execute(
                            insert(Batch).values(
                                points_count=st.session_state["city_count"]
                            )
                        ).inserted_primary_key[0]

                        for clustering_length in CLUSTERING_LENGTHS:
                            for rollback_max in MAX_ROLLBACK_CALCUL_MIN_PATH:
                                n_clustering = 1

                                for clustering_type, params in CLUSTERING_CONFIGURATIONS.items():
                                    with st.spinner(f"Calcul des clusters de taille {clustering_length} - {rollback_max} : {n_clustering} / {len(CLUSTERING_CONFIGURATIONS.items())}", show_time=True):

                                        param_names = list(params.keys())
                                        param_values_lists = list(params.values())
                                        all_combinations = list(itertools.product(*param_values_lists))

                                        n_combination = 1

                                        for combination_values in all_combinations:
                                            df_points_coords = df_points.copy()
                                            df_points_distances = df_distances_pairs.copy()
                                            iteration_params = dict(zip(param_names, combination_values))

                                            str_params = f" avec comme paramètres {iteration_params}" if len(param_names) > 0 else ''
                                            with st.spinner(f"Cluster {clustering_type}{str_params} : {n_combination} / {len(all_combinations)}", show_time=True):

                                                configuration_id = conn.execute(
                                                    insert(Configuration).values(
                                                        batch_id=batch_id,
                                                        clustering_length=clustering_length,
                                                        clustering_rollback=rollback_max,
                                                        clustering_type=clustering_type,
                                                        clustering_params=iteration_params,
                                                    )
                                                ).inserted_primary_key[0]

                                                layers_count = 0

                                                while True:
                                                    with st.spinner(f"Génération de la couche {layers_count}", show_time=True):
                                                        if clustering_type == 'kmeans':
                                                            df_points_coords = clustering.calcul_clusters_kmeans(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'gaussian':
                                                            df_points_coords = clustering.calcul_clusters_gaussian(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'dbscan':
                                                            df_points_coords = clustering.calcul_clusters_dbscan(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'dbscan':
                                                            df_points_coords = clustering.calcul_clusters_dbscan(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'hdbscan':
                                                            df_points_coords = clustering.calcul_clusters_hdbscan(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'affinity':
                                                            df_points_coords = clustering.calcul_clusters_affinity(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'meanshift':
                                                            df_points_coords = clustering.calcul_clusters_meanshift(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'spectral':
                                                            df_points_coords = clustering.calcul_clusters_spectral(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'agglomerative':
                                                            df_points_coords = clustering.calcul_clusters_agglomerative(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'optics':
                                                            df_points_coords = clustering.calcul_clusters_optics(df_points_coords, clustering_length, iteration_params)
                                                        elif clustering_type == 'birch':
                                                            df_points_coords = clustering.calcul_clusters_birch(df_points_coords, clustering_length, iteration_params)

                                                        # st.text(f"{clustering_type} - {layers_count}")
                                                        # st.dataframe(df_points_coords)

                                                        save_layer(
                                                            conn,
                                                            batch_id,
                                                            df_points_coords,
                                                            df_points_distances,
                                                            level=layers_count,
                                                            configuration_id=configuration_id,
                                                        )

                                                        if df_points_coords.shape[0] > clustering_length:
                                                            df_points_coords = clustering.calcul_centroids(df_points_coords)
                                                            df_points_coords, df_points_distances = calcul_distances(df_points_coords)

                                                            layers_count += 1
                                                        else:
                                                            break
                                            n_combination += 1
                                    n_clustering += 1

                    conn.commit()

                    with st.spinner(f"Calcul des chemins", show_time=True):
                        configurations = session.scalars(
                            select(Configuration) \
                                .filter(Configuration.batch_id == batch_id) \
                                .order_by(Configuration.id)
                        ).all()

                        n_configuration = 1

                        for configuration in configurations:
                            with st.spinner(f"{configuration!r} - Calcul {n_configuration} / {len(configurations)}", show_time=True):
                                layers = session.scalars(
                                    select(Layer)\
                                    .filter(Layer.configuration_id == configuration.id)\
                                    .order_by(Layer.level)
                                ).all()

                                df_centroids = None
                                n_layer = 1

                                for layer in layers[::-1]:
                                    with st.spinner(f"Calcul de la couche {n_layer} / {len(layers)}", show_time=True):
                                        df_cluster = None
                                        df_points_coords = layer.get_points()
                                        df_points_distances = get_distances(layer)

                                        if df_centroids is None:
                                            df_centroids = clustering.calcul_centroids(
                                                df_points_coords,
                                            )

                                        if df_centroids.shape[0] < BEST_PATH_MAX_CLUSTERS:
                                            # Le meilleur chemin est calculé par rapport aux groupes précédents
                                            df_cluster = best_path(df_centroids, df_points_coords, df_points_distances)

                                        if df_cluster is None:
                                            df_cluster = sort_points_by_cluster(df_centroids, df_points_coords)

                                        # Le min local est calculé indépedemment des groupes
                                        df_cluster = calcul_min_local_path(df_cluster, df_points_distances, rollback_max)

                                        for _, row_point in df_cluster.iterrows():
                                            point = session.query(Point) \
                                                .filter(Point.layer_id == layer.id) \
                                                .filter(Point.code == row_point['code']) \
                                                .update({
                                                'code_next' : row_point['code_next'],
                                                'lat_next'  : row_point['lat_next'],
                                                'lng_next'  : row_point['lng_next'],
                                                'distance'  : row_point['distance'],
                                            })

                                        df_centroids = df_cluster[['code', 'lat', 'lng', 'x', 'y', 'cluster']]

                                    n_layer += 1

                                configuration.distance_min = float(df_cluster['distance'].sum())

                            n_configuration += 1

                        session.commit()

                    if batch_id is not None:
                        goto_page_object('batch', 'batch_id', batch_id)

                except Exception as e:
                    session.rollback()
                    raise e


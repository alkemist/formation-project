import itertools

import streamlit as st
import pandas as pd

from formation.data.constants import BEST_PATH_MAX_CLUSTERS
from formation.helpers.clustering import precalcul_clusters, calcul_clusters_kmeans, calcul_clusters_gaussian, \
    calcul_centroids
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

                        clustering_lengths = [3]
                        configurations = {
                            'kmeans': {},
                            'gaussian': {
                                'covariance_type': ['full', 'tied', 'diag', 'spherical']
                            }
                        }

                        with st.spinner(f"Calcul des clusters", show_time=True):
                            for clustering_length in clustering_lengths:
                                with st.spinner(f"Taille de cluster {clustering_length}", show_time=True):
                                    for clustering_type, params in configurations.items():
                                        param_names = list(params.keys())
                                        param_values_lists = list(params.values())
                                        all_combinations = list(itertools.product(*param_values_lists))

                                        n_combination = 1

                                        for combination_values in all_combinations:
                                            df_points_coords = df_points.copy()
                                            df_points_distances = df_distances_pairs.copy()
                                            iteration_params = dict(zip(param_names, combination_values))

                                            with st.spinner(f"Type de cluster {clustering_type} avec comme paramètres {iteration_params} {n_combination} / {len(all_combinations)}", show_time=True):

                                                configuration_id = conn.execute(
                                                    insert(Configuration).values(
                                                        batch_id=batch_id,
                                                        clustering_length=clustering_length,
                                                        clustering_type=clustering_type,
                                                        clustering_params=iteration_params,
                                                    )
                                                ).inserted_primary_key[0]

                                                layers_count = 0

                                                while True:
                                                    with st.spinner(f"Génération de la couche {layers_count}", show_time=True):
                                                        if clustering_type == 'kmeans':
                                                            df_points_coords = calcul_clusters_kmeans(df_points_coords, clustering_length)
                                                        elif clustering_type == 'gaussian':
                                                            df_points_coords = calcul_clusters_gaussian(df_points_coords, clustering_length, iteration_params)

                                                        save_layer(
                                                            conn,
                                                            batch_id,
                                                            df_points_coords,
                                                            df_points_distances,
                                                            level=layers_count,
                                                            configuration_id=configuration_id,
                                                        )

                                                        if df_points_coords.shape[0] > clustering_length:
                                                            df_points_coords = calcul_centroids(df_points_coords)
                                                            df_points_coords, df_points_distances = calcul_distances(df_points_coords)

                                                            layers_count += 1
                                                        else:
                                                            break
                                            n_combination += 1

                            conn.commit()

                        with st.spinner(f"Calcul des chemins", show_time=True):
                            configurations = session.scalars(
                                select(Configuration) \
                                    .filter(Configuration.batch_id == batch_id) \
                                    .order_by(Configuration.id)
                            ).all()

                            for configuration in configurations:
                                layers = session.scalars(
                                    select(Layer)\
                                    .filter(Layer.configuration_id == configuration.id)\
                                    .order_by(Layer.level)
                                ).all()

                                df_centroids = None
                                n_layer = 1

                                for layer in layers[::-1]:
                                    with st.spinner(f"{configuration!r} - Calcul de la couche {n_layer} / {len(layers)}", show_time=True):
                                        df_cluster = None
                                        df_points_coords = layer.get_points()
                                        df_points_distances = get_distances(layer)

                                        if df_centroids is None:
                                            df_centroids = calcul_centroids(
                                                df_points_coords,
                                            )

                                        if df_centroids.shape[0] < BEST_PATH_MAX_CLUSTERS:
                                            # Le meilleur chemin est calculé par rapport aux groupes précédents
                                            df_cluster = best_path(df_centroids, df_points_coords, df_points_distances)

                                        if df_cluster is None:
                                            df_cluster = sort_points_by_cluster(df_centroids, df_points_coords)

                                        # Le min local est calculé indépedemment des groupes
                                        df_cluster = calcul_min_local_path(df_cluster, df_points_distances)

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

                            session.commit()

                    if batch_id is not None:
                        goto_page_object('batch', 'batch_id', batch_id)

                except Exception as e:
                    session.rollback()
                    raise e


import streamlit as st
import folium

from shapely import MultiPoint, centroid
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

from formation.data.pages import list_pages
from formation.data.map import CENTER_START, ZOOM_START
from formation.helpers.map import calcul_distance, map_center, map_reset, map_init, random_cities

st.title(list_pages['try_2']['title'])
st.set_page_config(
    page_title=list_pages['try_2']['title'],
    page_icon=list_pages['try_2']['icon'],
    layout="wide",
)

map_init()

min_lat = 42
max_lat = 51.5
min_lon = -5
max_lon = 9

colors = px.colors.qualitative.Vivid

top_left, top_middle, top_right = st.columns([2, 3, 2])

clustering_algo = None

def draw():
    st.session_state["markers_city"] = []

    for index, row in st.session_state["cities"].iterrows():
        st.session_state["markers_city"].append(
            folium.CircleMarker(
                location=[
                    row['lat'],
                    row['lng'],
                ],
                tooltip=f"{index} ({row['cluster']})",
                popup=f"{row['name']}",
                fill=True,
                color=colors[int(row['cluster'])],
                weight=0,
                fill_opacity=0.6,
                radius=15,
            )
        )

with top_left:
    st.write('Nombre de villes :')

    form_left, form_right = st.columns([1, 1], vertical_alignment='center')

    city_count = form_left.slider(
        "Nombre de ville :",
        value=st.session_state["city_count"],
        min_value=3,
        max_value=70,
        step=1,
        label_visibility='collapsed'
    )

    b1, b2, b3 = form_right.columns([2, 1, 1], vertical_alignment='center')

    if b1.button("Calcul", type="primary", use_container_width=True, key='calcul_cities'):
        if city_count != st.session_state["city_count"]:
            map_reset()
            st.session_state["city_count"] = city_count

            # Initialisation du random
            st.session_state["cities"], \
            st.session_state["distances"],= random_cities(city_count)

        map_center()

    if b2.button(label="", type="tertiary", icon=":material/replay:"):
        map_reset()

    if b3.button(label="", type="tertiary", icon=":material/home:"):
        map_center()

    if st.session_state["cities"] is not None:
        measures = {
            'kmeans': [],
            'gaussian': []
        }
        n_clusters_list = range(
            1,
            min(
                st.session_state["cities"].shape[0],
                len(colors) - 1
            ) + 1
        )

        points = st.session_state["cities"].loc[:,['x', 'y']].to_numpy()

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
        top_left.plotly_chart(fig, key='fig_clusters')

        draw()

with st.expander("Paramètre et compare les algorithmes", expanded=False):
    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        if st.session_state["cities"] is not None:
            points = st.session_state["cities"].loc[:,['x', 'y']].to_numpy()

            bottom_left_left, bottom_left_middle, bottom_left_right = bottom_left.columns([1, 3, 1], vertical_alignment='center')
            bottom_left_left.text('K-means')

            n_clusters_kmeans= bottom_left_middle.slider(
                "Nombre de clusters :",
                min_value=n_clusters_list[0],
                max_value=n_clusters_list[-1],
                step=1,
                label_visibility='collapsed',
                key='k_kmeans',
            )

            model_kmeans = KMeans(n_clusters=n_clusters_kmeans)
            model_kmeans.fit(points)
            st.session_state["cities"]['cluster_kmeans'] = model_kmeans.predict(points)
            st.session_state["cities"]['cluster_kmeans'] = st.session_state["cities"]['cluster_kmeans'].astype(str)
            st.session_state["n_clusters_kmeans"] = n_clusters_kmeans

            fig_kmeans = px.scatter(
                st.session_state["cities"],
                x='x',
                y='y',
                color='cluster_kmeans',
                hover_name='code',
                height=700,
                size=[1] * st.session_state["cities"].shape[0],
                color_discrete_sequence=colors,
            )

            fig_kmeans.update_yaxes(scaleanchor="x", scaleratio=1)
            bottom_left.plotly_chart(fig_kmeans, use_container_width=True, key='fig_kmeans')

    with bottom_right:
        if st.session_state["cities"] is not None:
            points = st.session_state["cities"].loc[:,['x', 'y']].to_numpy()

            bottom_right_left, bottom_right_middle, bottom_right_right = bottom_right.columns([1, 3, 1], vertical_alignment='center')
            bottom_right_left.text('Gaussian Mixture')

            n_clusters_gaussian = bottom_right_middle.slider(
                "Nombre de clusters :",
                min_value=n_clusters_list[0],
                max_value=n_clusters_list[-1],
                step=1,
                label_visibility='collapsed',
                key='k_gaussian',
            )

            model_gaussian = GaussianMixture(n_components=n_clusters_gaussian, random_state=42)
            model_gaussian.fit(points)
            st.session_state["cities"]['cluster_gaussian'] = model_gaussian.predict(points)
            st.session_state["cities"]['cluster_gaussian'] = st.session_state["cities"]['cluster_gaussian'].astype(str)
            st.session_state["n_clusters_gaussian"] = n_clusters_gaussian

            fig_gaussian = px.scatter(
                st.session_state["cities"],
                x='x',
                y='y',
                color='cluster_gaussian',
                hover_name='code',
                height=700,
                size=[1] * st.session_state["cities"].shape[0],
                color_discrete_sequence=colors,
            )

            fig_gaussian.update_yaxes(scaleanchor="x", scaleratio=1)
            bottom_right.plotly_chart(fig_gaussian, use_container_width=True, key='fig_gaussian')

with top_middle:
    if st.session_state["cities"] is not None:
        middle_left, middle_right = st.columns([1, 1], vertical_alignment='center')
        middle_left.text("Selection de l'algo :")
        clustering_algo = middle_right.selectbox(
            "",
            ("kmeans", "gaussian"),
            label_visibility='collapsed'
        )

with top_left:
    if st.session_state["cities"] is not None:
        if clustering_algo != "":
            k = st.session_state["n_clusters_" + clustering_algo]

            st.session_state["cities"]["cluster"] = st.session_state["cities"]["cluster_" + clustering_algo]

            draw()

            df_centroids = pd.DataFrame({
                'cluster': range(0, k),
                'lat': [0] * k,
                'lng': [0] * k
            })

            df_centroids_distances = pd.DataFrame()

            for i in df_centroids.index:
                points = st.session_state["cities"][st.session_state["cities"]['cluster'] == str(i)].loc[:, ['lat', 'lng']].to_numpy()
                c = centroid(MultiPoint(points))
                df_centroids.loc[i:i, 'lat'] = c.x
                df_centroids.loc[i:i, 'lng'] = c.y

            st.session_state["markers_center"] = []

            for index_a, row_a in df_centroids.iterrows():
                st.session_state["markers_center"].append(
                    folium.CircleMarker(
                        location=[
                            row_a['lat'],
                            row_a['lng'],
                        ],
                        tooltip=str(index_a),
                        fill=False,
                        color=colors[int(index_a)],
                        weight=5,
                        opacity=1,
                        radius=10,
                    )
                )

                for index_b, row_b in df_centroids.iterrows():
                    df_centroids_distances = pd.concat([
                        df_centroids_distances,
                        pd.DataFrame(
                            {
                                'cluster_a': [index_a],
                                'cluster_b': [index_b],
                                'distance': [
                                    calcul_distance(
                                        row_a['lat'],
                                        row_a['lng'],
                                        row_b['lat'],
                                        row_b['lng'],
                                    )
                                ],
                            }
                        )
                    ])

            df_centroids_distances = df_centroids_distances.rename(columns={'cluster_a': 'cluster'}) \
                .pivot(index='cluster', columns='cluster_b', values='distance')

            st.session_state["centers"] = df_centroids

with top_middle:
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

    for marker in st.session_state["markers_city"]:
        fg.add_child(marker)

    for marker in st.session_state["markers_center"]:
        fg.add_child(marker)

    st_folium(
        m,
        center=st.session_state["center"],
        zoom=st.session_state["zoom"],
        width='100%',
        feature_group_to_add=fg,
        returned_objects=['center', 'zoom'],
        key="map",
    )

with top_right:
    # if st.session_state["cities"] is not None:
    #     st.dataframe(st.session_state["cities"])
    # if st.session_state["distances"] is not None:
    #     st.dataframe(st.session_state["distances"])
    if st.session_state["cities"] is not None:
        st.dataframe(
            st.session_state["cities"] \
                .sort_values(['cluster', 'code']) \
                .set_index('code')
            [['name', 'cluster']]
        )

    if st.session_state["centers"] is not None:
        st.dataframe(st.session_state["centers"].set_index('cluster'))
import streamlit as st
import pandas as pd

from formation.data.pages import list_pages
from formation.data.map import DF_CITIES
from formation.helpers.map import calcul_min_path, \
    calcul_distances, calcul_centroids, map_draw, draw_markers, map_start, map_clear, draw_scores, draw_kmeans, \
    draw_gaussian, draw_centroids

st.title(list_pages['try_3']['title'])
st.set_page_config(
    page_title=list_pages['try_3']['title'],
    page_icon=list_pages['try_3']['icon'],
    layout="wide",
)

st.subheader(f'Etape 1')

top_left, top_middle, top_right = st.columns([1, 2, 2])

df_points = pd.DataFrame()
df_centroids = pd.DataFrame()

with top_left:
    st.write('Nombre de villes :')

    form_left, form_right = st.columns([1, 1], vertical_alignment='center')

    city_count = form_left.slider(
        "Nombre de ville :",
        min_value=3,
        max_value=70,
        step=1,
        label_visibility='collapsed'
    )

    b1, b2, b3 = form_right.columns([2, 1, 1], vertical_alignment='center')

    if b1.button("Calcul", type="primary", use_container_width=True):
        df_cities_random = DF_CITIES.sample(city_count)
        map_clear(0)

        st.session_state[f"df_points_0"], df_points_distances = calcul_distances(df_cities_random)
        draw_markers(0, st.session_state[f"df_points_0"])

    if b2.button(label="", type="tertiary", icon=":material/replay:"):
        map_clear(0)

    if b3.button(label="", type="tertiary", icon=":material/home:"):
        map_start(0)

    if f"df_points_0" in st.session_state and st.session_state[f"df_points_0"] is not None:
        top_left.plotly_chart(
            draw_scores(st.session_state[f"df_points_0"]),
            key='fig_clusters_0'
        )

if f"df_points_0" in st.session_state and st.session_state[f"df_points_0"] is not None:
    with st.expander("Paramètre et compare les algorithmes", expanded=True):
        bottom_left, bottom_right = st.columns([1, 1])

        draw_kmeans(0, bottom_left)
        draw_gaussian(0, bottom_right)

with top_middle:
    if f"df_points_0" in st.session_state and st.session_state[f"df_points_0"] is not None:
        middle_left, middle_right = st.columns([1, 1], vertical_alignment='center')
        middle_left.text("Selection de l'algo :")
        st.session_state['clustering_algo_0'] = middle_right.selectbox(
            "",
            ("kmeans", "gaussian"),
            label_visibility='collapsed',
            key=f'select_algo_0'
        )

        if f"n_clusters_{st.session_state['clustering_algo_0']}_0" in st.session_state:
            st.session_state[f"df_points_0"]["cluster"] = st.session_state[f"df_points_0"][f"cluster_{st.session_state['clustering_algo_0']}"]

        draw_markers(0, st.session_state[f"df_points_0"])

with top_right:
    if f"df_points_0" in st.session_state and st.session_state[f"df_points_0"] is not None:
        st.text('Villes')
        st.dataframe(st.session_state[f"df_points_0"])

        st.text('Centres')
        st.session_state[f"df_centroids_0"], df_centroids_distances = calcul_centroids(st.session_state[f"df_points_0"])
        st.dataframe(st.session_state[f"df_centroids_0"])

        draw_centroids(0, st.session_state[f"df_centroids_0"])

    # st.text('Meilleur chemin')
    # df_points_best = calcul_min_path(df_points, df_points_distances)
    # st.dataframe(df_points_best)



    # if df_centroids_distances.shape[0] > 0:
    #     df_centroids_best = calcul_min_path(df_centroids, df_centroids_distances)
    #     st.dataframe(df_centroids_best)

with top_middle:
    map_draw(0)


i = 1


if st.session_state[f"df_centroids_{i - 1}"] is not None:
    while st.session_state[f"df_centroids_{i - 1}"] is not None and st.session_state[f"df_centroids_{i - 1}"].shape[0] > 1:
        st.subheader(f'Etape {i + 1}')

        st.session_state[f"df_points_{i}"] = st.session_state[f"df_centroids_{i - 1}"]

        left, middle, right = st.columns([1, 2, 2])

        with left:
            left.plotly_chart(
                draw_scores(st.session_state[f"df_points_{i}"]),
                key=f'fig_clusters_{i}'
            )

        with st.expander("Paramètre et compare les algorithmes", expanded=True):
            bottom_left, bottom_right = st.columns([1, 1])

            draw_kmeans(i, bottom_left)
            draw_gaussian(i, bottom_right)

        with middle:
            middle_left, middle_right = st.columns([1, 1], vertical_alignment='center')
            middle_left.text("Selection de l'algo :")
            st.session_state['clustering_algo_{i}'] = middle_right.selectbox(
                "",
                ("kmeans", "gaussian"),
                label_visibility='collapsed',
                key=f'select_algo_{i}'
            )

            if f"n_clusters_{st.session_state['clustering_algo_{i}']}_{i}" in st.session_state:
                st.session_state[f"df_points_{i}"]["cluster"] = st.session_state[f"df_points_{i}"][f"cluster_{st.session_state['clustering_algo_{i}']}"]

            draw_markers(0, st.session_state[f"df_points_0"])

        with middle:
            map_draw(i)
            draw_markers(i, st.session_state[f"df_points_{i}"])

        with right:
            st.text('Villes')
            st.dataframe(st.session_state[f"df_points_{i}"])

            st.text('Centres')
            st.session_state[f"df_centroids_{i}"], df_centroids_distances = calcul_centroids(st.session_state[f"df_points_{i}"])
            st.dataframe(st.session_state[f"df_centroids_{i}"])

            draw_centroids(0, st.session_state[f"df_centroids_{i}"])

        i += 1

#st.json(st.session_state)
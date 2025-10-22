# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from sklearn.manifold import MDS
# import folium
# import random
# from shapely import MultiPoint, centroid
# from sklearn.preprocessing import StandardScaler
# from streamlit.delta_generator import DeltaGenerator
# from streamlit_folium import st_folium
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
#
#
#
# def map_center():
#     st.session_state["zoom"] = ZOOM_START + random.uniform(0.00001, 0.00009)
#     st.session_state["center"] = [
#         CENTER_START[0] + random.uniform(0.00001, 0.00009),
#         CENTER_START[1] + random.uniform(0.00001, 0.00009),
#         ]
#
# def draw_markers(map_index: int, df_points: pd.DataFrame):
#     st.session_state[f"markers_point_{map_index}"] = []
#
#     for index, row in df_points.iterrows():
#         st.session_state[f"markers_point_{map_index}"].append(
#             folium.CircleMarker(
#                 location=[
#                     row['lat'],
#                     row['lng'],
#                 ],
#                 tooltip=f"{index} ({row['cluster']})",
#                 fill=True,
#                 color=COLORS[int(row['cluster'])],
#                 weight=0,
#                 fill_opacity=0.6,
#                 radius=15,
#             )
#         )
#
#         if 'next' in df_points.columns:
#             city_next = df_points.loc[row['next']:row['next']]
#
#             st.session_state[f"markers_line_{map_index}"].append(
#                 folium.PolyLine(
#                     locations=[
#                         [row['lat'], row['lng']],
#                         [city_next['lat'].values[0], city_next['lng'].values[0]],
#                     ],
#                     color="#000000",
#                     weight=3,
#                 )
#             )
#
# def calcul_centroids(df_points: pd.DataFrame, reset_clusters = True):
#     clusters = df_points['cluster'].unique()
#     k = len(clusters)
#
#     df_centroids = pd.DataFrame({
#         'lat': [0] * k,
#         'lng': [0] * k
#     }, index=clusters)
#
#     df_centroids['lat'] = df_centroids['lat'].astype(float)
#     df_centroids['lng'] = df_centroids['lng'].astype(float)
#
#     for i in clusters:
#         points = df_points[df_points['cluster'] == str(i)].loc[:, ['lat', 'lng']].to_numpy()
#         c = centroid(MultiPoint(points))
#         df_centroids.loc[i:i, 'lat'] = c.x
#         df_centroids.loc[i:i, 'lng'] = c.y
#
#     return calcul_distances(df_centroids, reset_clusters = True)
#
# def draw_centroids(map_index: int, df_centers: pd.DataFrame):
#     st.session_state[f"markers_center_{map_index}"] = []
#
#     for index, row in df_centers.iterrows():
#         st.session_state[f"markers_center_{map_index}"].append(
#             folium.CircleMarker(
#                 location=[
#                     row['lat'],
#                     row['lng'],
#                 ],
#                 tooltip=str(index),
#                 fill=False,
#                 color=COLORS[int(index)],
#                 weight=5,
#                 opacity=1,
#                 radius=10,
#             )
#         )
#
# def draw_scores(df_points: pd.DataFrame):
#     measures = {
#         'kmeans': [],
#         'gaussian': []
#     }
#
#     n_clusters_list = range(
#         1,
#         min(
#             df_points.shape[0],
#             len(COLORS) - 1
#         ) + 1
#     )
#
#     points = df_points.loc[:,['x', 'y']].to_numpy()
#
#     for k in n_clusters_list:
#         kmeans = KMeans(n_clusters=k)
#         gaussian = GaussianMixture(n_components=k, random_state=42)
#
#         kmeans.fit(points)
#         gaussian.fit(points)
#
#         measures['kmeans'].append(kmeans.inertia_)
#         measures['gaussian'].append(gaussian.bic(points))
#
#     measures['kmeans'] = StandardScaler().fit_transform(np.array(measures['kmeans']).reshape(-1, 1)).reshape(-1)
#     measures['gaussian'] = StandardScaler().fit_transform(np.array(measures['gaussian']).reshape(-1, 1)).reshape(-1)
#
#     fig = px.line(
#         pd.DataFrame(measures, index=n_clusters_list),
#         title='Méthode du Coude',
#         labels={
#             'index': 'Nombre de Clusters',
#             'value': "Score",
#         },
#     )
#     fig.update_layout(showlegend=True)
#
#     return fig
#
# def map_prepare(map_index: int):
#
#     if f"center_{map_index}" not in st.session_state:
#         st.session_state[f"center_{map_index}"] = CENTER_START
#     if f"zoom_{map_index}" not in st.session_state:
#         st.session_state[f"zoom_{map_index}"] = ZOOM_START
#
#     if f"markers_point_{map_index}" not in st.session_state:
#         st.session_state[f"markers_point_{map_index}"] = []
#     if f"markers_center_{map_index}" not in st.session_state:
#         st.session_state[f"markers_center_{map_index}"] = []
#     if f"markers_line_{map_index}" not in st.session_state:
#         st.session_state[f"markers_line_{map_index}"] = []
#
#     if f"clustering_algo_{map_index}" not in st.session_state:
#         st.session_state[f"clustering_algo_{map_index}"] = 'kmeans'
#     if f"n_clusters_kmeans_{map_index}" not in st.session_state:
#         st.session_state[f"n_clusters_kmeans_{map_index}"] = 1
#     if f"n_clusters_gaussian_{map_index}" not in st.session_state:
#         st.session_state[f"n_clusters_gaussian_{map_index}"] = 1
#
#     if f"df_points_{map_index}" not in st.session_state:
#         st.session_state[f"df_points_{map_index}"] = None
#     if f"df_centroids_{map_index}" not in st.session_state:
#         st.session_state[f"df_centroids_{map_index}"] = None
#
# def map_clear(map_index: int):
#     map_start(map_index)
#
#     st.session_state[f"markers_point_{map_index}"] = []
#     st.session_state[f"markers_center_{map_index}"] = []
#     st.session_state[f"markers_line_{map_index}"] = []
#
#     st.session_state[f"clustering_algo_{map_index}"] = 'kmeans'
#     st.session_state[f"n_clusters_kmeans_{map_index}"] = 1
#     st.session_state[f"n_clusters_gaussian_{map_index}"] = 1
#
#     st.session_state[f"df_points_{map_index}"] = None
#     st.session_state[f"df_centroids_{map_index}"] = None
#
#
# def map_start(map_index: int):
#     st.session_state[f"zoom_{map_index}"] = ZOOM_START + random.uniform(0.00001, 0.00009)
#     st.session_state[f"center_{map_index}"] = [
#         CENTER_START[0] + random.uniform(0.00001, 0.00009),
#         CENTER_START[1] + random.uniform(0.00001, 0.00009),
#         ]
#
# def map_draw(map_index: int = 0):
#     m = folium.Map(
#         location=CENTER_START,
#         zoom_start=ZOOM_START,
#         min_zoom=ZOOM_START,
#         max_bounds=True,
#         min_lat=MIN_LAT,
#         max_lat=MAX_LAT,
#         min_lon=MIN_LNG,
#         max_lon=MAX_LNG,
#     )
#
#     fg = folium.FeatureGroup(name="Markers")
#
#     folium.PolyLine(
#         locations=[
#             [MIN_LAT, MIN_LNG],
#             [MAX_LAT, MIN_LNG],
#             [MAX_LAT, MAX_LNG],
#             [MIN_LAT, MAX_LNG],
#             [MIN_LAT, MIN_LNG],
#         ],
#         color="#FF0000",
#         weight=2,
#     ).add_to(fg)
#
#     for marker in st.session_state[f"markers_point_{map_index}"]:
#         fg.add_child(marker)
#
#     for marker in st.session_state[f"markers_center_{map_index}"]:
#         fg.add_child(marker)
#
#     for marker in st.session_state[f"markers_line_{map_index}"]:
#         fg.add_child(marker)
#
#     st_folium(
#         m,
#         center=st.session_state[f"center_{map_index}"],
#         zoom=st.session_state[f"zoom_{map_index}"],
#         width='100%',
#         feature_group_to_add=fg,
#         returned_objects=['center', 'zoom'],
#         key=f"map_{map_index}",
#     )
#
# def prepare_clustering(map_index: int, el: DeltaGenerator, algo_type: str):
#     n_clusters_list = range(
#         1,
#         min(
#             st.session_state[f"df_points_{map_index}"].shape[0],
#             len(COLORS) - 1
#         ) + 1
#     )
#
#     k = el.slider(
#         f"Nombre de clusters {algo_type} : ",
#         min_value=n_clusters_list[0],
#         max_value=n_clusters_list[-1],
#         step=1,
#         #label_visibility='collapsed',
#         key=f'k_{algo_type}_{map_index}',
#     )
#
#     return st.session_state[f"df_points_{map_index}"].loc[:,['x', 'y']].to_numpy(), k
#
# def finish_clustering(map_index: int, el: DeltaGenerator, algo_type: str):
#     st.session_state[f"df_points_{map_index}"][f'cluster_{algo_type}'] = \
#         st.session_state[f"df_points_{map_index}"][f'cluster_{algo_type}'].astype(str)
#
#     fig = px.scatter(
#         st.session_state[f"df_points_{map_index}"].reset_index(),
#         x='x',
#         y='y',
#         color=f'cluster_{algo_type}',
#         hover_name='index',
#         height=700,
#         size=[1] * st.session_state[f"df_points_{map_index}"].shape[0],
#         color_discrete_sequence=COLORS,
#     )
#
#     fig.update_yaxes(scaleanchor="x", scaleratio=1)
#
#     el.plotly_chart(fig, use_container_width=True, key=f'fig_{algo_type}_{map_index}')
#
# def draw_kmeans(map_index: int, el: DeltaGenerator):
#     # el.text('K-means')
#
#     points, k_kmeans = prepare_clustering(
#         map_index,
#         el,
#         'kmeans',
#     )
#
#     if st.session_state[f"n_clusters_kmeans_{map_index}"] != k_kmeans:
#         model_kmeans = KMeans(n_clusters=k_kmeans)
#         model_kmeans.fit(points)
#         st.session_state[f"df_points_{map_index}"]['cluster_kmeans'] = model_kmeans.predict(points)
#
#         if st.session_state[f'clustering_algo_{map_index}'] == 'kmeans':
#             st.session_state[f"df_points_{map_index}"]["cluster"] = st.session_state[f"df_points_{map_index}"][f"cluster_kmeans"]
#             st.session_state[f"df_points_{map_index}"]["cluster"] = st.session_state[f"df_points_{map_index}"]["cluster"].astype(str)
#
#             st.session_state[f"df_centroids_{map_index}"], df_centroids_distances = calcul_centroids(st.session_state[f"df_points_{map_index}"])
#             st.session_state[f"n_clusters_kmeans_{map_index}"] = k_kmeans
#
#     finish_clustering(
#         map_index,
#         el,
#         'kmeans',
#     )
#
# def draw_gaussian(map_index: int, el: DeltaGenerator):
#     # el.text('Gaussian Mixture')
#
#     points, k_gaussian = prepare_clustering(
#         map_index,
#         el,
#         'gaussian',
#     )
#
#     if st.session_state[f"n_clusters_gaussian_{map_index}"] != k_gaussian:
#         model_gaussian = GaussianMixture(n_components=k_gaussian, random_state=42)
#         model_gaussian.fit(points)
#         st.session_state[f"df_points_{map_index}"]['cluster_gaussian'] = model_gaussian.predict(points)
#
#         if st.session_state[f'clustering_algo_{map_index}'] == 'gaussian':
#             st.session_state[f"df_points_{map_index}"]["cluster"] = st.session_state[f"df_points_{map_index}"][f"cluster_gaussian"]
#             st.session_state[f"df_points_{map_index}"]["cluster"] = st.session_state[f"df_points_{map_index}"]["cluster"].astype(str)
#
#             st.session_state[f"df_centroids_{map_index}"], df_centroids_distances = calcul_centroids(st.session_state[f"df_points_{map_index}"])
#             st.session_state[f"n_clusters_gaussian_{map_index}"] = k_gaussian
#
#     finish_clustering(
#         map_index,
#         el,
#         'gaussian',
#     )

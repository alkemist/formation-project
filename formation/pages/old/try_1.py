import streamlit as st
import folium
import plotly.express as px
from streamlit_folium import st_folium

from formation.data.pages import list_pages
from formation.data.map import CENTER_START, ZOOM_START
from formation.helpers.old.map import map_center, map_reset, map_init, random_cities

st.title(list_pages['try_1']['title'])
st.set_page_config(
    page_title=list_pages['try_1']['title'],
    page_icon=list_pages['try_1']['icon'],
    layout="wide",
)

map_init()

min_lat = 42
max_lat = 51.5
min_lon = -5
max_lon = 9

colors = px.colors.qualitative.Vivid

top_left, top_middle, top_right = st.columns([1, 2, 2])

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

    if b1.button("Calcul", type="primary", use_container_width=True):
        if city_count != st.session_state["city_count"]:
            map_reset()
            st.session_state["city_count"] = city_count

            st.session_state["cities"], \
            st.session_state["distances"],= random_cities(city_count)

        map_center()

        df_cities = st.session_state["cities"]

        df_distances = st.session_state["distances"]

        city_codes_random = df_cities['code'].unique()
        city_begined = []
        city_tours = {}

        while True:
            cities_not_first = list(
                filter(
                    lambda v: v not in city_begined,
                    city_codes_random
                )
            )

            if len(cities_not_first) > 0:
                city_first = cities_not_first[0]
                city_begined.append(city_first)

                city_added = [city_first]
                distance_total = 0
                while True:
                    city_code = city_added[-1]
                    distances = df_distances[city_code].to_dict()

                    cities_not_added = {
                        k: v
                        for k, v in distances.items()
                        if k not in city_added
                    }

                    if len(cities_not_added) > 0:
                        city_next = min(cities_not_added, key=lambda k: cities_not_added[k])
                        distance_total += min(cities_not_added.values())

                        city_added.append(city_next)
                    else:
                        break

                city_tours[city_first] = {
                    'distance': distance_total,
                    'cities': city_added
                }

            else:
                break

        best_tours = min(city_tours.values(), key=lambda obj: obj["distance"])['cities']

        df_cities["index"] = df_cities['code'].map({
            code: i
            for i, code in enumerate(best_tours)
        })

        df_cities = df_cities.sort_values(by='index')

        df_cities['next'] = df_cities['code'].shift(periods=-1, fill_value='')
        df_cities['next'] = df_cities['next'].astype(str)
        df_cities.loc[df_cities.index[-1], 'next'] = df_cities.iloc[0]['code']
        df_cities["distance"] = df_cities.apply(
            lambda r: df_distances.loc[r["code"], r["next"]] if r["next"] else 0,
            axis=1
        )
        df_cities["distance_cumul"] = df_cities["distance"].cumsum()

        st.session_state["city_tours"] = df_cities.set_index('code')

    if b2.button(label="", type="tertiary", icon=":material/replay:"):
        map_reset()

    if b3.button(label="", type="tertiary", icon=":material/home:"):
        map_center()

    if st.session_state["city_tours"] is not None:
        st.session_state["markers_city"] = []

        df_cities = st.session_state["city_tours"]

        for index, row in df_cities.iterrows():
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

            if row['next'] != '':
                city_next = df_cities.loc[row['next']:row['next']]

                st.session_state["markers_city"].append(
                    folium.PolyLine(
                        locations=[
                            [row['lat'], row['lng']],
                            [city_next['lat'].values[0], city_next['lng'].values[0]],
                        ],
                        color="#000000",
                        weight=3,
                    )
                )

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
    # if st.session_state["distances"] is not None:
    #     st.dataframe(st.session_state["distances"])

    if st.session_state["city_tours"] is not None:
        st.dataframe(
            st.session_state["city_tours"]\
                .sort_index()\
                .drop(columns=['x', 'y', 'cluster'])
        )
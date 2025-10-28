import random
from typing import Type, Callable

import pandas as pd
import streamlit as st
from sqlalchemy import select, delete
from sqlalchemy.exc import NoResultFound

from formation import session, Base, engine
from formation.data.map import CENTER_START, ZOOM_START
from formation.data.pages import list_pages

from folium import CircleMarker, PolyLine, Map, FeatureGroup
from shapely import MultiPoint, centroid


def redirect_with_error(page: str, e: Exception):
    st.session_state['error'] = e
    st.switch_page(list_pages[page]['link'])

def goto_page_object(page: str, object_key: str, object_id: int):
    st.session_state[object_key] = object_id
    st.switch_page(list_pages[page]['link'])

def get_page_object(class_: Type[Base], object_key, page_error):
    try:
        if object_key in st.query_params:
            object_id = int(st.query_params[object_key])
        elif object_key in st.session_state:
            object_id = int(st.session_state[object_key])
        else:
            raise KeyError('Aucun "{}" sélectionné'.format(object_key))

        return session.scalars(select(class_).where(class_.id == object_id)).one()

    except (KeyError, TypeError, ValueError, NoResultFound) as e:
        redirect_with_error(page_error, e)

def confirm_delete(class_: Type[Base], object_id, page_end):
    with engine.connect() as conn:
        conn.execute(
            delete(class_).filter(class_.id == object_id)
        )

        conn.commit()

    st.switch_page(list_pages[page_end]['link'])

def map_center():
    st.session_state["zoom"] = ZOOM_START + random.uniform(0.00001, 0.00009)
    st.session_state["center"] = [
        CENTER_START[0] + random.uniform(0.00001, 0.00009),
        CENTER_START[1] + random.uniform(0.00001, 0.00009),
    ]

def map_draw(df_points: pd.DataFrame, column_color: str|None = None, dict_colors_fill: dict|None = None, dict_colors_border: dict|None = None):
    m = Map(
        location=st.session_state["center"] if "center" in st.session_state else CENTER_START,
        zoom_start=st.session_state["zoom"] if "zoom" in st.session_state else ZOOM_START,
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
                    tooltip=f"{point['code']} ({point[column_color]})" if column_color is not None else point['code'],

                    # border
                    color=dict_colors_border[int(point[column_color])] if column_color is not None else 'blue',
                    # opacity=0.6,
                    # weight=5,

                    # fill
                    fill=True,
                    fill_color=dict_colors_fill[int(point[column_color])] if column_color is not None else 'blue',
                    fill_opacity=1,
                    radius=15,
                )
            )

        if point['lat_next'] is not None and point['lng_next'] is not None:
            fg.add_child(
                PolyLine(
                    locations=[
                        [point['lat'], point['lng']],
                        [point['lat_next'], point['lng_next']],
                    ],
                    color=dict_colors_fill[int(point[column_color])] if column_color is not None else 'blue',
                    fill_color=dict_colors_fill[int(point[column_color])] if column_color is not None else 'blue',
                    weight=4,
                )
            )

    m.add_child(fg)

    return m

def generate_colors(n: int) -> list[str]:
    hex_colors = []

    if n == 1:
        return ['blue']

    luminosity_step = 255 / (n - 1)

    # Define the range of values [0-255]
    value_range = range(n)

    for i in value_range:
        grayscale_value = round(i * luminosity_step)

        component_value = grayscale_value

        component_value = max(0, min(255, component_value))

        hex_component = f'{component_value:02X}'

        hex_code = f"#{hex_component}{hex_component}{hex_component}"

        hex_colors.append(hex_code)

    return hex_colors



@st.dialog("Confirmation")
def confirm(callback: Callable, class_: Type[Base], object_id, page_end):
    st.write(f"Etes vous sûr ?")
    button_left, button_right = st.columns([1,1])
    if button_left.button("Non"):
        st.rerun()

    if button_right.button("Oui"):
        callback(class_, object_id, page_end)
        st.rerun()
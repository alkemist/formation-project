import streamlit as st
from streamlit_folium import st_folium

from formation.data.pages import list_pages
from formation.helpers.map import get_distances
from formation.helpers.streamlit import map_center, map_draw
from formation.helpers.streamlit import get_page_object, goto_page_object
from formation.models.layer import Layer

layer = get_page_object(Layer, 'layer_id', 'batches')

st.session_state['layer_id'] = layer.id
st.session_state['batch_id'] = layer.batch_id

df_points = layer.get_points()

################################################################################################################""

title = str(layer)

st.title(title)
st.set_page_config(
    page_title=title,
    page_icon=list_pages['layer']['icon'],
    layout="wide",
)

title_left, title_middle, title_right = st.columns([1, 10, 1], vertical_alignment='bottom')
title_middle.subheader('Liste des points')

if title_left.button(icon=':material/arrow_back:', label='', type='tertiary'):
    if layer.configuration is None:
        goto_page_object('batch', 'batch_id', layer.batch_id)
    else:
        goto_page_object('configuration', 'configuration_id', layer.configuration_id)

if title_right.button(icon=':material/my_location:', label='', type='tertiary'):
    map_center()

col_left, col_middle, col_right = st.columns([4, 2, 6])

with col_left:
    # df_distances = get_distances(layer)
    # st.dataframe(df_distances)
    pass

col_middle.dataframe(
    df_points[['code', 'lat', 'lng']],
    column_config={
        "code": st.column_config.TextColumn(
            'Code',
            pinned=True,
        ),
    },
    use_container_width=True,
    hide_index=True,
    height=700,
)

with col_right:
    st_folium(map_draw(df_points), width='100%')
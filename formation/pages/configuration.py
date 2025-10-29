import pandas as pd
import streamlit as st
from sqlalchemy import select, delete, or_, and_
from sqlalchemy.orm import Session
from streamlit import switch_page
from streamlit_folium import st_folium

from formation import engine
from formation.data.pages import list_pages
from formation.helpers.streamlit import get_page_object, goto_page_object, map_draw, map_center, \
    confirm, confirm_delete, generate_colors
from formation.models import Batch, Point, Layer, Configuration

configuration = get_page_object(Configuration, 'configuration_id', 'batches')

st_points = select(Configuration, Layer, Point) \
    .filter(Layer.id == Point.layer_id) \
    .where(
        or_(
            and_(
                Configuration.id == configuration.id,
                Configuration.id == Layer.configuration_id,
            )
        )
    ) \
    .order_by(Layer.level)

df_points = pd.read_sql(st_points, engine) \
    [['id', 'layer_id', 'level', 'code', 'lat', 'lng', 'code_next', 'lat_next', 'lng_next']]

df_layers = df_points[['layer_id', 'level', 'id']].groupby(['layer_id', 'level']).count().reset_index() \
    .rename(columns={'id': 'count'}) \
    .rename(columns={'layer_id': 'id'})
df_layers['link'] = '/layer?layer_id=' + df_layers['id'].astype(str)

################################################################################################################""

title = str(configuration)

st.title(title)
st.set_page_config(
    page_title=title,
    page_icon=list_pages['configuration']['icon'],
    layout="wide",
)

title_left, title_middle, title_action_1, title_action_2 = st.columns([1, 9, 1, 1], vertical_alignment='bottom')
title_middle.subheader('Liste des couches')

if title_left.button(icon=':material/arrow_back:', label='', type='tertiary'):
    goto_page_object('batch', 'batch_id', configuration.batch_id)

if title_action_1.button(icon=':material/delete:', label='', type='tertiary'):
    confirm(confirm_delete, Batch, configuration.id, 'batches')

if title_action_2.button(icon=':material/my_location:', label='', type='tertiary'):
    map_center()

col_left, col_right = st.columns([6, 6])

table_event = col_left.dataframe(
    df_layers[['level', 'count', 'link']],
    column_config={
        "level": st.column_config.NumberColumn(
            'Couche',
            pinned=True,
        ),
        "count": st.column_config.NumberColumn(
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
    use_container_width=True,
    hide_index=True,
    selection_mode='multi-row',
    on_select="rerun",
)

with col_right:
    df_layers_filtered = df_layers.iloc[table_event.selection.rows]
    df_points_filtered = df_points.loc[df_points['layer_id'].isin(df_layers_filtered['id'].unique())]

    color_indexes = df_layers['level'].unique()
    color_values = generate_colors(len(color_indexes))
    dict_colors_fill = dict(zip(color_indexes, color_values))
    dict_colors_border = dict(zip(color_indexes, color_values[::-1]))

    st_folium(map_draw(df_points_filtered, 'level', dict_colors_fill, dict_colors_border), width='100%')
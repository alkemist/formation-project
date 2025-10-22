import pandas as pd
import streamlit as st
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from streamlit import switch_page
from streamlit_folium import st_folium

from formation import engine
from formation.data.pages import list_pages
from formation.helpers.streamlit import redirect_with_error, get_page_object, goto_page_object, map_draw, map_center, \
    confirm, confirm_delete
from formation.models import Batch, Point, Layer

batch = get_page_object(Batch, 'batch_id', 'batches')

st_points = select(Batch, Layer, Point) \
    .filter(Batch.id == Layer.batch_id) \
    .filter(Layer.id == Point.layer_id) \
    .filter(Batch.id == batch.id) \
    .order_by(Layer.id)

df_points = pd.read_sql(st_points, engine) \
    [['id', 'layer_id', 'level', 'code', 'lat', 'lng']]

df_layers = df_points[['layer_id', 'level', 'id']].groupby(['layer_id', 'level']).count().reset_index()\
    .rename(columns={'id': 'count'})\
    .rename(columns={'layer_id': 'id'})
df_layers['link'] = '/layer?layer_id=' + df_layers['id'].astype(str)

################################################################################################################""

title = str(batch)

st.title(title)
st.set_page_config(
    page_title=title,
    page_icon=list_pages['batch']['icon'],
    layout="wide",
)

title_left, title_middle, title_action_1, title_action_2 = st.columns([1, 9, 1, 1], vertical_alignment='bottom')
title_middle.subheader('Liste des couches')

if title_left.button(icon=':material/arrow_back:', label='', type='tertiary'):
    switch_page(list_pages['batches']['link'])

if title_action_1.button(icon=':material/delete:', label='', type='tertiary'):
    confirm(confirm_delete, Batch, batch.id, 'batches')

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

table_event.selection.rows = [1]

with col_right:
    df_layers_filtered = df_layers.iloc[table_event.selection.rows]
    df_points_filtered = df_points.loc[df_points['layer_id'].isin(df_layers_filtered['id'].unique())]
    st_folium(map_draw(df_points_filtered, 'level'), width='100%')
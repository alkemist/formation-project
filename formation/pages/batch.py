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
from formation.models import Batch, Point, Layer, Configuration

batch = get_page_object(Batch, 'batch_id', 'batches')

st_configurations = select(Batch, Configuration, Layer) \
    .filter(Batch.id == Configuration.batch_id) \
    .filter(Configuration.id == Layer.configuration_id) \
    .filter(Batch.id == batch.id) \
    .order_by(Configuration.id)

df_configurations = pd.read_sql(st_configurations, engine)\
    [['clustering_type', 'clustering_length', 'clustering_rollback', 'clustering_params', 'distance_min', 'id_1', 'id_2']]\
    .groupby(['id_1']) \
    .agg(
        clustering_type=('clustering_type', 'first'),
        clustering_length=('clustering_length', 'first'),
        clustering_rollback=('clustering_rollback', 'first'),
        clustering_params=('clustering_params', 'first'),
        distance_min=('distance_min', 'first'),
        count=('id_2', 'nunique'),
).reset_index().rename(columns={'id_1': 'id'})

df_configurations['link'] = '/configuration?configuration_id=' + df_configurations['id'].astype(str)

################################################################################################################""

title = str(batch)

st.title(title)
st.set_page_config(
    page_title=title,
    page_icon=list_pages['batch']['icon'],
    layout="wide",
)

title_left, title_middle, title_action_1, title_action_2 = st.columns([1, 9, 1, 1], vertical_alignment='bottom')
title_middle.subheader('Liste des configurations')

if title_left.button(icon=':material/arrow_back:', label='', type='tertiary'):
    switch_page(list_pages['batches']['link'])

if title_action_1.button(icon=':material/delete:', label='', type='tertiary'):
    confirm(confirm_delete, Batch, batch.id, 'batches')

if title_action_2.button(icon=':material/my_location:', label='', type='tertiary'):
    map_center()

table_event = st.dataframe(
    df_configurations.drop(columns=['id']),
    column_config={
        "clustering_type": st.column_config.TextColumn(
            'Type',
            pinned=True,
        ),
        "clustering_length": st.column_config.NumberColumn(
            'Longueur',
        ),
        "clustering_rollback": st.column_config.NumberColumn(
            'Rollbacks',
        ),
        "clustering_params": st.column_config.JsonColumn(
            'Paramètres',
        ),
        "distance_min": st.column_config.NumberColumn(
            'Distance',
        ),
        "count": st.column_config.NumberColumn(
            'Couches',
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
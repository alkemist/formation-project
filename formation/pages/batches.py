import streamlit as st
import pandas as pd
from sqlalchemy import select, insert

from formation import engine
from formation.data.pages import list_pages
from formation.helpers.map import calcul_distances
from formation.helpers.streamlit import goto_page_object
from formation.models import Batch, Point, Layer

st.title(list_pages['batches']['title'])
st.set_page_config(
    page_title=list_pages['batches']['title'],
    page_icon=list_pages['batches']['icon'],
    layout="wide",
)

INPUT_DIR = 'data/formated/'

tab_list, tab_create = st.tabs(["Liste des lots", "Créer un lot"])

with tab_list:
    st_batches = select(Batch).order_by(Batch.id)
    df_batches = pd.read_sql(st_batches, engine)
    df_batches['link'] = '/batch?batch_id=' + df_batches['id'].astype(str)

    st.dataframe(
        df_batches,
        column_config={
            "id": st.column_config.Column(
                'Id',
                pinned=True,
            ),
            "link": st.column_config.LinkColumn(
                "Liste",
                max_chars=100,
                display_text=":material/visibility:",
                pinned=False,
                width=100,
            ),
        },
        column_order=('id', 'link'),
        selection_mode='single-row',
        hide_index=True,
    )

batch = None

with tab_create:
    top_left, top_middle, top_right = st.columns([2, 1, 2])

    top_left.text("Créer un lot avec")

    city_count = st.session_state["city_count"] if "city_count" in st.session_state else 3

    st.session_state["city_count"] = top_middle.number_input("", value=city_count, min_value=3, max_value=255, step=1, label_visibility='collapsed', width='stretch')

    if top_right.button("Selectionner les villes", type="primary", use_container_width=True):
        with st.spinner("Génération de la 1er couche", show_time=True):
            df_cities = pd.read_csv(INPUT_DIR + 'cities-france.csv',
                                    dtype={'code_insee': str, 'dep_code': str, 'nom_standard': str, 'latitude_mairie': float, 'longitude_mairie': float},
                                    ).rename(columns={'code_insee': 'code', 'nom_standard': 'name', 'latitude_mairie': 'lat', 'longitude_mairie': 'lng'}) \
                .set_index('code') \
                .loc[:, ['lat', 'lng']]
            df_points, df_distances = calcul_distances(df_cities.sample(st.session_state["city_count"]))

            st.session_state["points"] = df_points

    if "points" in st.session_state and st.session_state is not None:
        st.dataframe(
            st.session_state["points"],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Créer le lot", type="primary", use_container_width=True):
            with engine.connect() as conn:
                batch_id = conn.execute(
                    insert(Batch).values()
                ).inserted_primary_key[0]

                layer_id = conn.execute(
                    insert(Layer).values(
                        batch_id=batch_id,
                        level=0,
                    )
                ).inserted_primary_key[0]

                for _, point in st.session_state["points"].iterrows():
                    st_point = insert(Point).values(
                        layer_id=layer_id,
                        code=point['code'],
                        lat=point['lat'],
                        lng=point['lng'],
                        x=point['x'],
                        y=point['y'],
                    )
                    conn.execute(st_point)

                conn.commit()

            if batch_id is not None:
                del st.session_state["points"]
                goto_page_object('batch', 'batch_id', batch_id)



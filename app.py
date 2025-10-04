import streamlit as st
from formation.data.pages import list_pages

st.header('Projet de formation')

st.set_page_config(layout="wide")

st.html("""<style>
    .stMain > .stMainBlockContainer{
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
</style>""")

pg = st.navigation(
    [ st.Page('formation/pages/home.py', title='Accueil', icon="🏠") ]
    + [
        st.Page(page['link'], title=page['title'], icon=page['icon']) for page in list_pages.values()
    ],
    position="top"
)
pg.run()
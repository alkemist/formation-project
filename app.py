import streamlit as st

st.header('Projet de formation')

from formation.data.pages import list_pages

pg = st.navigation(
    [ st.Page('formation/pages/home.py', title='Accueil', icon="🏠") ]
    + [
        st.Page(page['link'], title=page['title'], icon=page['icon']) for page in list_pages.values()
    ],
    position="top"
)
pg.run()
import streamlit as st

from formation.data.pages import list_pages

st.title('Optimisation de trajets')
st.header('Fonctionnalités')

for page in list_pages.values():
    st.page_link(page['link'], label=page['description'], icon=page['icon'])
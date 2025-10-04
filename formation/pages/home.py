import streamlit as st

from formation.data.pages import list_pages

st.title('Liste des pages')

for page in list_pages.values():
    st.page_link(page['link'], label=page['title'], icon=page['icon'])
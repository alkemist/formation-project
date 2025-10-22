import streamlit as st

from formation.data.pages import list_pages
from formation.database import engine
from formation import Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)

    st.set_page_config(layout="wide")

    st.html("""<style>
        .stMain > .stMainBlockContainer{
            padding-top: 4rem;
            padding-bottom: 2rem;
        }
        .stMarkdownBadge {width: 100%}
    </style>""")

    pg = st.navigation(
        [ st.Page('formation/pages/home.py', title='Accueil', icon="🏠") ]
        + [
            st.Page(page['link'], title=page['title'], icon=page['icon']) for page in list_pages.values()
        ],
        position="top"
    )

    if 'error' in st.session_state:
        st.error(str(st.session_state['error'])[1:-1], width='stretch')
        del st.session_state['error']

    pg.run()
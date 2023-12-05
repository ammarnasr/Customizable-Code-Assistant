import streamlit as st
from repo_search_utils.streamlit_app import fetch_repos_app, extract_code_app, browse_repo_info_app


st.title("Customizable Code Assistant")

st.markdown(
    """
    This app retrieves the most starred Github repos based on the search query and filters you provide.
    """
)

tab1, tab2, tab3 = st.tabs(["Search Repos", "Extract Code", "Browse Repos"])

with tab1:
    fetch_repos_app(0)

with tab2:
    extract_code_app(100)

with tab3:
    browse_repo_info_app(200)
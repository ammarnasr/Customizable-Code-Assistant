import streamlit as st
from repo_search_utils.streamlit_app import fetch_repos_app, extract_code_app, browse_repo_info_app, upload_dataset_hugging_face


st.title("Customizable Code Assistant")

st.markdown(
    """
    This app retrieves the most starred Github repos based on the search query and filters you provide.
    """
)

# tab1, tab2, tab3, tab4 = st.tabs(["Search Repos", "Extract Code", "Browse Repos", "Upload Dataset to Hugging Face"])
# tab1 = st.tabs(["Search Repos"])

# with tab1:
#     # st.write("")
    # fetch_repos_app(0)
# with tab2:
#     # st.write("")
#     extract_code_app(100)

# with tab3:
#     # st.write("")
#     browse_repo_info_app(200)

# with tab4:
#     st.write("")
#     upload_dataset_hugging_face(300)

fetch_repos_app(0)
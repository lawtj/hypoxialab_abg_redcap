import streamlit as st


st.set_page_config(page_title="ABL Upload", layout="wide")

st.title("ABL Upload")
st.write("This app is now split into separate flows so UCSF and Uganda can evolve independently.")
st.markdown(
    """
    Use the Streamlit sidebar to choose a page:

    1. `UCSF`: restored to the last known working UCSF flow.
    2. `Uganda`: separate uploader for the diverged Uganda CSV format.
    """
)
st.info("Open a page from the sidebar to begin.")

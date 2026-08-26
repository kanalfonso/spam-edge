import streamlit as st

scam_explorer_page = st.Page(
    "nodes/scam_explorer.py", 
    title="Scam Explorer", 
    icon="🕵"
)


variant_generator_page = st.Page(
    "nodes/variant_generator.py", 
    title="Variant Generator", 
    icon="👾"
)


pg = st.navigation(
    {
        "Nodes": [
            scam_explorer_page, 
            variant_generator_page
        ]
    }
)

pg.run()
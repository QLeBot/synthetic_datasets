import pandas as pd
import streamlit as st

from src.coords_generator import (
    random_points_on_land_fast,
    random_points_in_country_equal_area,
)


st.set_page_config(page_title="Synthetic points map", layout="wide")
st.title("Synthetic points map")

tab_generate, tab_load = st.tabs(["Generate", "Load CSV"])

with tab_generate:
    st.subheader("Generate points")

    mode = st.radio("Mode", ["On land (global)", "Inside a country"], horizontal=True)
    n = st.number_input("Number of points", min_value=10, max_value=500_000, value=10_000, step=1000)
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000_000, value=42, step=1)

    if mode == "Inside a country":
        country = st.text_input("Country name (Natural Earth 'name')", value="Germany")
    else:
        country = None

    if st.button("Generate", type="primary"):
        with st.spinner("Generating points..."):
            if mode == "Inside a country":
                pts = random_points_in_country_equal_area(str(country), int(n), seed=int(seed))
            else:
                pts = random_points_on_land_fast(int(n), seed=int(seed))

        # pts is Nx2 [lon, lat]
        df = pd.DataFrame(pts, columns=["lon", "lat"])

        st.success(f"Generated {len(df):,} points")
        st.map(df, latitude="lat", longitude="lon")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="points.csv",
            mime="text/csv",
        )

with tab_load:
    st.subheader("Load points from CSV")
    st.caption("Expecting columns named `lat`/`lon` (preferred) or `latitude`/`longitude`.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        lat_col = cols.get("lat") or cols.get("latitude")
        lon_col = cols.get("lon") or cols.get("longitude")

        if not lat_col or not lon_col:
            st.error(f"Couldn't find lat/lon columns. Columns found: {list(df.columns)}")
        else:
            view = df.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]].dropna()
            st.write(f"Loaded {len(view):,} points")
            st.map(view, latitude="lat", longitude="lon")


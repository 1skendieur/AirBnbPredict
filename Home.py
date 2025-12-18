import streamlit as st
import pandas as pd

from src.predict import load_model, predict_price_one

st.set_page_config(page_title="London Airbnb Price Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed.csv")

@st.cache_resource
def get_model():
    return load_model()

df = load_data()
model = get_model()

st.title("London Airbnb Price Predictor")
st.caption("Predict nightly price based on location and listing attributes.")


st.sidebar.header("Map filters")
room_types = sorted(df["room_type"].dropna().unique())
selected_room_types = st.sidebar.multiselect(
    "Room type(s)",
    options=room_types,
    default=room_types
)


use_neigh_filter = st.sidebar.checkbox("Filter by neighbourhood", value=False)
if use_neigh_filter:
    neighbourhoods = sorted(df["neighbourhood"].dropna().unique())
    selected_neigh = st.sidebar.selectbox("Neighbourhood", neighbourhoods)
    df_map_src = df[(df["room_type"].isin(selected_room_types)) & (df["neighbourhood"] == selected_neigh)]
else:
    df_map_src = df[df["room_type"].isin(selected_room_types)]

df_map_src = df_map_src.dropna(subset=["latitude","longitude"])
n_points = min(len(df_map_src), 4000)
df_map = df_map_src.sample(n_points, random_state=42).rename(columns={"latitude":"lat","longitude":"lon"})

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Listings map")
    st.map(df_map[["lat","lon"]])

    st.caption(f"Showing {n_points} points (sampled)")

with col2:
    st.subheader("Price prediction")


    default_lat = float(df["latitude"].median())
    default_lon = float(df["longitude"].median())


    neighbourhoods = sorted(df["neighbourhood"].dropna().unique())

    with st.form("predict_form"):
        neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
        room_type = st.selectbox("Room type", room_types)

        latitude = st.number_input("Latitude", value=default_lat, format="%.6f")
        longitude = st.number_input("Longitude", value=default_lon, format="%.6f")

        minimum_nights = st.number_input("Minimum nights", min_value=1, value=2, step=1)
        number_of_reviews = st.number_input("Number of reviews", min_value=0, value=10, step=1)
        reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=0.5, step=0.1)
        calculated_host_listings_count = st.number_input("Host listings count", min_value=1, value=1, step=1)
        availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=180, step=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        x = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "room_type": str(room_type),
            "neighbourhood": str(neighbourhood),
            "minimum_nights": int(minimum_nights),
            "number_of_reviews": int(number_of_reviews),
            "reviews_per_month": float(reviews_per_month),
            "calculated_host_listings_count": int(calculated_host_listings_count),
            "availability_365": int(availability_365),
        }

        pred = predict_price_one(x, model=model)

        st.success(f"Predicted nightly price: **£{pred:.2f}**")


        st.caption("Predicted location point:")
        st.map(pd.DataFrame([{"lat": x["latitude"], "lon": x["longitude"]}]))
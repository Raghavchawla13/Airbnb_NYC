import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import base64
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/HP/Desktop/Raghav Everything/college stuff/Airbnb/AB_NYC_2019.csv")
    df = df.drop(columns=["id", "name", "host_name", "last_review", "neighbourhood"], errors='ignore')
    df.fillna(0, inplace=True)
    return df

@st.cache_resource
def perform_clustering(df, n_clusters=5):
    features = df[[ 
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365"
    ]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    df["cluster"] = cluster_labels

    # Calculate silhouette score
    silhouette = silhouette_score(scaled_features, cluster_labels)

    return df, kmeans, scaler, silhouette

def predict_cluster(scaler, kmeans, user_input_df):
    scaled_input = scaler.transform(user_input_df)
    cluster = kmeans.predict(scaled_input)[0]
    return cluster

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def plot_clusters(df):
    fig = px.scatter_mapbox(df, 
                            lat="latitude", lon="longitude",
                            color="cluster",
                            hover_name="price",
                            mapbox_style="carto-positron",
                            zoom=10,
                            height=600)
    st.plotly_chart(fig)

def show_reviews():
    st.markdown("### 📝 User Reviews")
    reviews = [
        {"name": "John", "review": "Amazing place, great host! ⭐⭐⭐⭐⭐"},
        {"name": "Emily", "review": "Loved the decor and cleanliness. ⭐⭐⭐⭐"},
        {"name": "Raj", "review": "Perfect for a weekend getaway. ⭐⭐⭐⭐⭐"}
    ]
    for r in reviews:
        st.write(f"**{r['name']}**: {r['review']}")

def show_contact():
    st.markdown("### 📞 Contact Us")
    st.markdown("""
    - 📧 Email: Raghavsupport@bnbapp.com  
    - 📱 Phone: +1-234-567-890
    """)

def main():
    st.set_page_config(page_title="FindMyFlat NYC", layout="wide")
    set_background(r"C:\Users\HP\Pictures\airbnb 1.jpg")

    st.title(":house: FindMyFlat NYC")

    df = load_data()
    df, kmeans, scaler, silhouette = perform_clustering(df)

    st.subheader(":wrench: Set your preferences:")

    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=40.75)
        longitude = st.number_input("Longitude", value=-73.98)
        min_nights = st.slider("Minimum Nights", 1, 30, 3)
        num_reviews = st.slider("Number of Members", 0, 500, 10)
    with col2:
        reviews_per_month = st.slider("Reviews per Month", 0.0, 10.0, 1.0)
        listings_count = st.slider("BedRoom Count", 1, 50, 1)
        availability = st.slider("Availability (days/year)", 0, 365, 150)

    user_input = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": min_nights,
        "number_of_reviews": num_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": listings_count,
        "availability_365": availability
    }])

    if st.button("🔍 Predict"):
        cluster = predict_cluster(scaler, kmeans, user_input)
        st.success(f"🎯 Your listing is in **Cluster #{cluster}**")

        avg_price = df[df["cluster"] == cluster]["price"].mean()
        st.info(f"💰 Estimated average price: **${avg_price:.2f}**")

        st.subheader(":world_map: Listings in Your Cluster")
        plot_clusters(df[df["cluster"] == cluster])

        st.dataframe(df[df["cluster"] == cluster][["latitude", "longitude", "price"]].head(10))

    st.markdown("### 📊 Clustering Quality")
    st.markdown(f"**Silhouette Score**: {silhouette:.4f} _(range: -1 to 1)_")

    show_reviews()
    show_contact()

if __name__ == "__main__":
    main()

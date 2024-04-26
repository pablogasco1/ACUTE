import streamlit as st
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import osmnx as ox
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.spatial.distance import pdist, squareform

token = "pk.eyJ1IjoicGFibG9nYXNjbyIsImEiOiJjbHN0MHloeGwweGwzMmtxaXFobDNwc29tIn0.cXi2ESWBKoAPywLexpqruQ"


def calculate_distance(df):
    # Radius of the Earth in kilometers
    r = 6371.0

    # Convert degrees to radians
    lat1 = np.radians(df['latitude'])
    lon1 = np.radians(df['longitude'])
    lat2 = np.radians(df['home_lat'])
    lon2 = np.radians(df['home_lon'])

    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers between the points on the sphere
    distance_sphere = r * c

    # Difference in elevation between the points
    delev = df['elevation']

    # Euclidean distance
    distance = np.sqrt(distance_sphere**2 + delev**2)

    return distance

##########################################################################
# Create a file uploader
uploaded_files = st.file_uploader("Choose Parquet files", type="parquet", accept_multiple_files=True)

dfs = []

# Check if any files have been uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read each parquet file into a DataFrame
        df_i = pd.read_parquet(uploaded_file)
        dfs.append(df_i)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)

        # Create date and time selectors
        start_time = st.date_input('Start date', value=pd.to_datetime('2023-06-21'))
        end_time = st.date_input('End date', value=pd.to_datetime('2023-12-31'))
        
        # Convert the date objects to datetime
        start_time = datetime.combine(start_time, datetime.min.time())
        end_time = datetime.combine(end_time, datetime.min.time())

        # Create a dropdown menu for station names
        station_names = df['station_name'].unique().tolist()
        station_names.insert(0, station_names.pop(station_names.index("0QRDKC2R03J32P")))
        selected_station = st.selectbox('Select a station', station_names)

        # Apply the mask
        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = (df['timestamp'] > start_time) & (df['timestamp'] <= end_time) & (df["station_name"] == selected_station) & (df["latitude"] > 3)
        df = df[mask] 
        df['distance'] = df.apply(calculate_distance, axis=1)

        #st.write(df)
        
# Load the dataframe outside the function
#data_dir = Path("/data/pgcasado/datasets/acute/")
#df = pd.concat(
#   pd.read_parquet(parquet_file)
#   for parquet_file in data_dir.glob('*.parquet')
#)

# Define your function
def plot_clusters(cluster_mode, altitude_limit, eps, min_samples, selected_tags):  
    # Use the global df inside the function
    global df  
    mask_alt = (df["altitude"] > altitude_limit)
    df_filtered_mask = df.loc[mask_alt].copy()
    lat_min, lat_max = df_filtered_mask["latitude"].min(), df_filtered_mask["latitude"].max()
    lon_min, lon_max = df_filtered_mask["longitude"].min(), df_filtered_mask["longitude"].max()
    polygon = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
    #tags = {"building":["university",]}
    print("****************", selected_tags, "**********************")
    #tags = {tag.split(": ")[0]: [tag.split(": ")[1]] for tag in selected_tags}
    # Initialize an empty dictionary
    tags = {}

    # Loop through the selected tags
    for tag in selected_tags:
        # Split the tag into category and value
        category, value = tag.split(": ")

        # If the category is not in the dictionary, add it with an empty list
        if category not in tags:
            tags[category] = []

        # Append the value to the list of values for this category
        tags[category].append(value)
    print("****************", tags, "**********************")
    dfx = ox.features.features_from_polygon(polygon, tags= tags).copy()  # Create a copy to avoid SettingWithCopyWarning
    dfx.dropna(subset=['name'], inplace=True)
    dfx['longitude'] = dfx['geometry'].apply(lambda x: x.centroid.coords.xy[0][0])
    dfx['latitude'] = dfx['geometry'].apply(lambda x: x.centroid.coords.xy[1][0])
    if not cluster_mode:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # Fit the model to the data
        dbscan.fit(df_filtered_mask[['latitude', 'longitude']])
        df_filtered_mask["cluster"] = dbscan.labels_
        dfn = df_filtered_mask[df_filtered_mask["cluster"] >= 0] # cluster -1 is those points with no cluster
        labels = np.delete(dbscan.labels_, np.where(dbscan.labels_ == -1))
    else:
        centroids = dfx[["latitude", "longitude"]].to_numpy()
        points = df_filtered_mask[["latitude", "longitude"]].to_numpy()
        distances = cdist(points, centroids, "euclidean")
        clusters = np.argmin(distances, axis=1)
        counts = np.bincount(clusters)
        N = 100
        large_clusters = np.where(counts > N)[0]
        # Filter out small clusters and their centroids
        filtered_centroids = centroids[large_clusters]
        distances = cdist(points, filtered_centroids, "euclidean")
        # Find the index of the closest centroid for each point
        clusters = np.argmin(distances, axis=1)
        df_filtered_mask["cluster"] = clusters
        dfn = df_filtered_mask[df_filtered_mask["cluster"] >= 0]
    # Assuming df1 and df2 are your dataframes
    fig = px.scatter_mapbox(dfn, lat='latitude', lon='longitude', color='cluster',
                            color_continuous_scale=px.colors.cyclical.HSV,
                        zoom=10, height=1000, width=1500,
                        title='Drone Detected Position',)

    # Add scatter_mapbox trace for df2
    if not cluster_mode:
        fig.add_trace(go.Scattermapbox(
            lat=dfx['latitude'],
            lon=dfx['longitude'],
            mode='markers',
            marker=dict(size=10, color='white'),
            name=""
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=dfx['latitude'].iloc[large_clusters],
            lon=dfx['longitude'].iloc[large_clusters],
            mode='markers',
            marker=dict(size=10, color='white'),
            name=""
        ))

    fig.update_layout(
        mapbox_style="dark",
        mapbox_accesstoken=token,
        margin={"r":0,"t":0,"l":0,"b":0},
        autosize=True,
        hovermode='closest',
        showlegend=True
    )
    
    cluster_df = pd.DataFrame()
    for i in range(df_filtered_mask["cluster"].max()):
        df_i = df_filtered_mask[df_filtered_mask["cluster"]==i]
        cluster_df.loc[i, "nº detection"] = df_i.shape[0]
        cluster_df.loc[i, "max_altitude"] = df_i["altitude"].max()
        cluster_df.loc[i, "std_altitude"] = round(df_i["altitude"].std(), 2)
        cluster_df.loc[i, "mean_altitude"] = round(df_i["altitude"].mean(), 2)
        cluster_df.loc[i, "max_distance"] = round(df_i["distance"].max(),2)
        cluster_df.loc[i, "std_distance"] = round(df_i["distance"].std(),2)
        cluster_df.loc[i, "mean_distance"] = round(df_i["distance"].mean(),2)
        cluster_df.loc[i, "main model"] = df_i["model"].value_counts().idxmax()
        cluster_df.loc[i, "nº model"] = len(df_i["model"].unique())
        cluster_df.loc[i, "nº ID"] = len(df_i["ident"].unique())
        cluster_df.loc[i, "nº days"] = pd.to_datetime(df_i['timestamp']).dt.date.nunique()

    cluster_df["nº detection"] = cluster_df["nº detection"].astype("int")
    cluster_df["nº ID"] = cluster_df["nº ID"].astype("int")
    cluster_df["nº days"] = cluster_df["nº days"].astype("int")
    cluster_df["nº model"] = cluster_df["nº model"].astype("int")
    st.write(cluster_df)
    st.plotly_chart(fig)

# Use Streamlit widgets to get user input
cluster_mode = st.checkbox('POI')
altitude_limit = st.slider('Altitude Limit', min_value=0, max_value=500, value=100, step=10)
eps = st.slider('Eps', min_value=0.001, max_value=0.2, value=0.025, step=0.001)
min_samples = st.slider('Min Samples', min_value=1, max_value=50, value=17)
selected_tags = st.multiselect('Tags', options=["building: church", "building: chapel", "building: university", "building: hospital", "building: government", "building: stadium", "amenities: prison", "amenities: police"], default=["building: university"])

# Call the function when the 'Plot Clusters' button is clicked
if st.button('Plot Clusters'):
    plot_clusters(cluster_mode, altitude_limit, eps, min_samples, selected_tags)


import streamlit as st
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import osmnx as ox
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS    
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from pathlib import Path
from io import BytesIO
import tempfile
import xlsxwriter

token = "pk.eyJ1IjoicGFibG9nYXNjbyIsImEiOiJjbHN0MHloeGwweGwzMmtxaXFobDNwc29tIn0.cXi2ESWBKoAPywLexpqruQ"

def cluster_dataframe(df_alt_filter):
    cluster_df = pd.DataFrame()
    for i in range(df_alt_filter["cluster"].min(), df_alt_filter["cluster"].max() + 1):
        df_i = df_alt_filter[df_alt_filter["cluster"]==i]
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
    
    return cluster_df

def merge_close_points(df_points_interest, R=0.01):
    # Calculate the distance between each point and each centroid
    centroids = df_points_interest[["latitude", "longitude"]].to_numpy()
    print(centroids)
    # Calculate the pairwise distances between all centroids
    distances_centroids = squareform(pdist(centroids))

    # Initialize a list to hold the new centroids
    new_centroids = []
    
    # Initialize a set to keep track of used centroids
    used_centroids = set()

    # For each centroid
    for i in range(len(centroids)):
        # Skip this centroid if it has already been used
        if tuple(centroids[i]) in used_centroids:
            continue

        # Find the other centroids that are closer than R distance
        close_centroids_indices = np.where(distances_centroids[i] <= R)[0]

        # Mark these centroids as used
        for index in close_centroids_indices:
            used_centroids.add(tuple(centroids[index]))

        # Calculate the mean latitude and longitude of these close centroids
        close_centroids = centroids[close_centroids_indices]
        mean_centroid = close_centroids.mean(axis=0)

        # Add the new centroid to the list
        new_centroids.append(mean_centroid)

    # Convert the list to a numpy array
    centroids_arr = np.array(new_centroids)
    
    return centroids_arr

def df_points_of_interest(df_alt_filter, selected_tags):
    lat_min, lat_max = df_alt_filter["latitude"].min(), df_alt_filter["latitude"].max()
    lon_min, lon_max = df_alt_filter["longitude"].min(), df_alt_filter["longitude"].max()
    polygon = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])

    tags_dict = dict()
    # Loop through the selected tags
    for tag in selected_tags:
        # Split the tag into category and value
        category, value = tag.split(": ")

        # If the category is not in the dictionary, add it with an empty list
        if category not in tags_dict:
            tags_dict[category] = []

        # Append the value to the list of values for this category
        tags_dict[category].append(value)
    df_points_interest = ox.features.features_from_polygon(polygon, tags= tags_dict).copy()  # Create a copy to avoid SettingWithCopyWarning
    df_points_interest.dropna(subset=['name'], inplace=True)
    df_points_interest['longitude'] = df_points_interest['geometry'].apply(lambda x: x.centroid.coords.xy[0][0])
    df_points_interest['latitude'] = df_points_interest['geometry'].apply(lambda x: x.centroid.coords.xy[1][0])
    
    #merged_centroids = merge_close_points(df_points_interest)
    
    return df_points_interest, tags_dict

def point_inside_polygon(point, polygon):
    # Create shapely Point and Polygon objects
    shapely_point = Point(point)
    shapely_polygon = Polygon(polygon)
    # Check if the point is inside the polygon
    inout = shapely_point.within(shapely_polygon)
    return inout

def scale_polygon(polygon, centroid, factor):
    scaled_polygon = []
    for point in polygon:
        scaled_point = centroid + factor * (point - centroid)
        scaled_polygon.append(scaled_point)
    return np.array(scaled_polygon)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_distance(df):
    # Radius of the Earth in kilometers
    r = 6371.0

    # Convert degrees to radians
    lat1 = np.radians(df['latitude'])
    lon1 = np.radians(df['longitude'])
    lat2 = np.radians(df['home_lat'])
    lon2 = np.radians(df['home_lon'])

    # Distance in kilometers between the points on the sphere
    distance_sphere = haversine(lat1, lon1, lat2, lon2)

    # Difference in elevation between the points
    delev = df['elevation']

    # Euclidean distance
    distance = np.sqrt(distance_sphere**2 + delev**2)

    return distance
    
def plot_point_of_interest(df, altitude_limit, min_samples, R, X, show_unclustered, selected_tags):
    

    alt_filter = (df["altitude"] > altitude_limit)
    df_alt_filter = df.loc[alt_filter].copy()
    df_alt_filter.reset_index(drop=True, inplace=True)
    
    df_points_interest, tags_dict = df_points_of_interest(df_alt_filter, selected_tags)

    centroids = merge_close_points(df_points_interest, R)
    print("centroids:", centroids)
    #print(centroids)
    # Calculate the distance between each point and each centroid
    # centroids = df_points_interest[["latitude", "longitude"]].to_numpy()

    # # Calculate the pairwise distances between all centroids
    # distances_centroids = squareform(pdist(centroids))

    # # Initialize a list to hold the new centroids
    # new_centroids = []

    # # For each centroid
    # for i in range(len(centroids)):
    #     # Find the other centroids that are closer than R distance
    #     close_centroids = centroids[distances_centroids[i] <= R]
        
    #     # Calculate the mean latitude and longitude of these close centroids
    #     mean_centroid = close_centroids.mean(axis=0)
        
    #     # Add the new centroid to the list
    #     new_centroids.append(mean_centroid)

    # # Convert the list to a numpy array
    # centroids = np.array(new_centroids)

    points = df_alt_filter[["latitude", "longitude"]].to_numpy()
    distances0 = cdist(points, centroids, "euclidean")

    # Find the index of the closest centroid for each point
    clusters0 = np.argmin(distances0, axis=1)
    print("clusters0:", clusters0)

    # Find the minimum distance for each point to its closest centroid and keep those with a closer distance than X
    min_distances = np.min(distances0, axis=1)
    dist_mask = min_distances <= X
    points = points[dist_mask]
    clusters0 = clusters0[dist_mask]
    df_reduced = df_alt_filter[dist_mask].copy()

    # Count the number of points in each cluster
    counts = np.bincount(clusters0)

    # Identify clusters with more than min_samples points
    large_clusters = np.where(counts > min_samples)[0]
    print("large_clusters:", large_clusters)

    # Filter out small clusters and their centroids
    filtered_centroids = centroids[large_clusters]

    # Recalculate the distances
    distances = cdist(points, filtered_centroids, "euclidean")

    # Find the index of the closest centroid for each point
    clusters = np.argmin(distances, axis=1)
    print("final clusters:", clusters)
    

    # Find the minimum distance for each point to its closest centroid and keep those with a closer distance than X
    min_distances = np.min(distances, axis=1)
    dist_mask = min_distances <= X
    points = points[dist_mask]
    clusters = clusters[dist_mask]
    df_reduced = df_reduced[dist_mask]
    df_reduced["cluster"] = clusters
    
    if show_unclustered:
        df_alt_filter = df_alt_filter.merge(df_reduced[["cluster"]], how='left', left_index=True, right_index=True)
        df_alt_filter['cluster'] = df_alt_filter['cluster'].fillna(-1)
        df_alt_filter["cluster"] = df_alt_filter["cluster"].astype(int)
    else:
        df_alt_filter = df_reduced
    
    parameters_dict = {"altitude_limit" : altitude_limit, "min_samples" : min_samples, "centroid_radius" : R, "min_distance" : X, "algorithm" : "POI", "tags" : tags_dict}
    
    #show_graph_table(df_alt_filter, pd.DataFrame(filtered_centroids, columns=["latitude", "longitude"]), tags_dict, parameters_dict)
    show_graph_table(df_alt_filter, df_points_interest.iloc[large_clusters], tags_dict, parameters_dict)

def plot_dbscan(df, altitude_limit, min_samples, eps, algorithm, show_unclustered, selected_tags):  
    alt_filter = (df["altitude"] > altitude_limit)
    df_alt_filter = df.loc[alt_filter].copy()
    df_points_interest, tags_dict = df_points_of_interest(df_alt_filter, selected_tags)
    
    if algorithm=="DBSCAN":
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm=="HDBSCAN":
        dbscan = HDBSCAN(cluster_selection_epsilon=eps, min_samples=min_samples)
    elif algorithm=="OPTICS":
        dbscan = OPTICS(max_eps=eps, min_samples=min_samples)
        
    # Fit the model to the data
    dbscan.fit(df_alt_filter[['latitude', 'longitude']])
    df_alt_filter["cluster"] = dbscan.labels_
    if not show_unclustered:
        df_alt_filter = df_alt_filter[df_alt_filter["cluster"] >= 0] # cluster -1 is those points with no cluster
    #labels = np.delete(dbscan.labels_, np.where(dbscan.labels_ == -1))
    
    parameters_dict = {"altitude_limit" : altitude_limit, "min_samples" : min_samples, "max_distance" : eps, "algorithm" : algorithm}
    
    show_graph_table(df_alt_filter, df_points_interest, tags_dict, parameters_dict)

def show_graph_table(df_alt_filter, df_points_interest, tags_dict, parameters_dict):
    # df_alt_filter is the dataframe for all the points with a cluster_id>0
    # df_points_interest is the dataframe that contains all the points of interest

    # Plot the drones points
    df_no_cluster = df_alt_filter[df_alt_filter["cluster"] == -1]
    df_cluster = df_alt_filter[df_alt_filter["cluster"] != -1]
    fig = px.scatter_mapbox(df_cluster, lat='latitude', lon='longitude', color='cluster', size="altitude",
                            color_continuous_scale=px.colors.diverging.Portland ,#px.colors.cyclical.HSV,
                            zoom=10, height=1000, width=1500,
                            title='Drone Detected Position',)
    
    # Now, the text with the number of the cluster is placed in the centroid of the cluster.
    centroids_list = list()
    for cluster_id, group_df in df_cluster.groupby('cluster'):
        centroid_lat = group_df['latitude'].mean()
        centroid_lon = group_df['longitude'].mean()
        
        centroids_list.append({
        'latitude': centroid_lat,
        'longitude': centroid_lon,
        'cluster': cluster_id
    })

    # Concatenate the list with 'centroids_df'
    centroids_df = pd.DataFrame(centroids_list)
    
    #Scatter points with no cluster
    #if not df_no_cluster.empty:
    if True:
        fig.add_trace(go.Scattermapbox(
            lat=df_no_cluster['latitude'],
            lon=df_no_cluster['longitude'],
            mode='markers',  # Include text labels
            marker=dict(size=5, color='black'),
            name="",
        ))
    else:
        distances = cdist(df_points_interest[["latitude", "longitude"]], centroids_df[["latitude", "longitude"]])
        closest_centroid_distances = np.min(distances, axis=1)
        max_dist = 0.03
        df_points_interest = df_points_interest[closest_centroid_distances < max_dist]
    
    # Plot 
    fig.add_trace(go.Scattermapbox(
        lat=centroids_df['latitude'],
        lon=centroids_df['longitude'],
        mode='text',  # Include text labels
        text=[str(i) for i in centroids_df["cluster"].values],
        textfont=dict(size=26, color='white'),
        name="",
    ))
    
    #Scatter points of interest
    fig.add_trace(go.Scattermapbox(
        lat=df_points_interest['latitude'],
        lon=df_points_interest['longitude'],
        mode='markers',  # Include text labels
        marker=dict(size=10, color='white'),
        name="",
    ))
    
    fig.update_layout(
        mapbox_style="dark",
        mapbox_accesstoken=token,
        margin={"r":0,"t":0,"l":0,"b":0},
        autosize=True,
        hovermode='closest',
        showlegend=True
    )
    
    # Compute convex hull for each cluster and plot
    df_points_interest['cluster_id'] = -1  # Initialize cluster ID column
    
    for cluster_id, group_df in df_cluster.groupby('cluster'):
        points = group_df[['latitude', 'longitude']].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the polygon
        
        # Create Polygon object for the cluster convex hull
        # Scale the polygon to make it bigger
        cluster_polygon = hull_points.tolist()
        centroid = np.mean(cluster_polygon, axis=0)
        scaled_polygon = scale_polygon(np.array(cluster_polygon), centroid, 1.1)  # Scale by 10%
        # Assign cluster ID to points of interest based on whether they are inside the cluster polygon
        df_points_interest.loc[df_points_interest.apply(
            lambda row: point_inside_polygon((row['latitude'], row['longitude']), scaled_polygon),
            axis=1
        ), 'cluster_id'] = cluster_id
            
        fig.add_trace(go.Scattermapbox(
            lat=scaled_polygon[:, 0],
            lon=scaled_polygon[:, 1],
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'Cluster {cluster_id}',
            showlegend=False  # Set showlegend to False for convex hull traces
        ))

    df_poi_cluster = df_points_interest[df_points_interest["cluster_id"] != -1]  
    # create and empty dataframe with an index until the last cluster with a POI inside, the rest will be dismissed.
    count_df = pd.DataFrame(index=range(df_poi_cluster["cluster_id"].max()+1))
    for tag in tags_dict:
        for elem in tags_dict[tag]:
            cond_i = df_poi_cluster[tag] == elem
            count_df[tag + "_" + elem] = df_poi_cluster[cond_i]["cluster_id"].value_counts()
    count_df = count_df.fillna(0)
    count_df.loc['Total', :] = count_df.sum(axis=0)
    count_df["Total"] = count_df.sum(axis=1)
    count_df = count_df[count_df['Total'] != 0] # finally remove those columns which have 0 POI inside

    cluster_df = cluster_dataframe(df_alt_filter)
    st.write(cluster_df)
    
    with BytesIO() as output:
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df_alt_filter.to_excel(writer, sheet_name='Data', index=False)
        df_parameters = pd.DataFrame.from_dict(parameters_dict, orient="index")
        df_parameters.T.to_excel(writer, sheet_name='Parameters', index=False)
        writer.close()
    
        st.download_button(
            label="Download Excel workbook",
            data=output.getvalue(),
            file_name="workbook.xlsx",
            mime="application/vnd.ms-excel"
        )
        
    st.plotly_chart(fig)
    st.write(count_df)

##########################################################################
# Create a file uploader
uploaded_files = st.file_uploader("Choose Parquet files", type="parquet", accept_multiple_files=True)
#uploaded_files = Path("/data/pgcasado/datasets/acute/")

dfs = []
df = pd.DataFrame()
grouped_df = pd.DataFrame()

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
        
        # Filter 1 based on the user selection
        time_mask = (df['timestamp'] > start_time) & (df['timestamp'] <= end_time) & (df["station_name"] == selected_station)
        df = df[time_mask]
        
        # Filter 2 to remove those points further than 1.5 times the std
        coordinates_df = df[['latitude', 'longitude']]
        z_scores = np.abs((coordinates_df - coordinates_df.mean()) / coordinates_df.std())
        threshold = 2
        mask = (z_scores < threshold).all(axis=1)
        df = df[mask]
        
        # Add a new column that will be used later
        df['distance'] = df.apply(calculate_distance, axis=1)

        grouped_df = df.groupby("journey").agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            altitude=("altitude", "max"),
            distance=("distance", "max"),
            journey=("journey", "first"),
            ident = ("ident", "first"),
            model = ("model", "first"), # shouldnt be the first, but the most used one
            timestamp = ("timestamp", "first"),
            )

        # for index, row in grouped_df.iterrows():
        #     centroid_coords = row[["latitude", "longitude"]].values.reshape(1, -1).astype(float)
        #     points = df[df["journey"]==index][["latitude", "longitude"]].values.astype(float)
        #     distances = cdist(centroid_coords, points, metric="euclidean")
        #     # Find the maximum distance for each centroid
        #     max_distance = distances.max(axis=1)
        #     grouped_df.loc[index, "radius"] = max_distance


# Use Streamlit widgets to get user input
#point_interest_mode = st.checkbox('POI')

algorithm = st.radio(
    "Cluster Algorithm:",
    ["DBSCAN", "HDBSCAN", "OPTICS", "POI"], 
    help = "Select the algorithm to cluster the flights: \
        DBSCAN: Density-Based Spatial Clustering of Applications with Noise. \
        Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density. \n \
        HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise. Performs DBSCAN over varying epsilon values \
        and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying \
        densities (unlike DBSCAN), and be more robust to parameter selection.\n \
        OPTIC: Ordering Points To Identify the Clustering Structure, closely related to DBSCAN, finds core sample of high density and expands \
        clusters from them [1]. Unlike DBSCAN, keeps cluster hierarchy for a variable neighborhood radius. Better suited for usage on large datasets \
        than the current sklearn implementation of DBSCAN. \n \
        POI: Point of Interest, cluster the points to the nearest point of interest selected by the user"
)

#st.select_slider('Slide to select', options=[1,'2'])
altitude_limit = st.slider('Altitude Limit', min_value=0, max_value=500, value=10, step=10,
                           help="Remove those flights with altitude lower than the limit")

if algorithm=="POI":
    R = st.slider('Centroid Radius', min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%f", 
                  help="Points of Interest inside the same radius will be averaged into one")
    X = st.slider('Min Distance', min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%f",
                  help="Points further than the minimum distance won't be considered for the cluster creation")
    #N = st.slider('Min Samples', min_value=1, max_value=150, value=17, step=1)
else:
    eps = st.slider('Max Distance', min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%f",
                    help="For DBSCAN stands for: The maximum distance between two samples for one to be considered as in the neighborhood of the other. \
                    This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately \
                    for your data set and distance function. \
                    For HDBSCAN stands for: A distance threshold. Clusters below this value will be merged. \
                    For OPTICS stands for: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")

min_samples = st.slider('Min Samples', min_value=1, max_value=100, value=17,
                        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.")

selected_tags = st.multiselect('Tags', options=["building: church", "building: chapel", "building: university", "building: hospital", 
                                                "building: government", "building: stadium", "amenity: prison", "amenity: police"], 
                               default=["building: university"],
                               help="Select which Point of Interest are going to be visualized")

# Call the function when the 'Plot Clusters' button is clicked

show_unclustered = st.checkbox('Show unclustered points', value=False, 
                          help="Show those points that haven't been assigned to a cluster")


reduce_jrny = st.checkbox('Reduce Journey', value=True, 
                          help="Consider only one point per journey, this point could represent the mean or the maximum values of all the points taken in a journey")

if reduce_jrny:
    mean_max = st.radio(
    "Max or Mean:",
    ["Max", "Mean"],
    help="The altidude and distance from the pilot considered to create the table will be the mean or the maximum of the journey. Latitude and longitude will always be the mean") 
    
    if not df.empty:
        if mean_max == "Mean":
            grouped_df = df.groupby("journey").agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                altitude=("altitude", "mean"),
                distance=("distance", "mean"),
                journey=("journey", "first"),
                ident = ("ident", "first"),
                model = ("model", "first"),
                timestamp = ("timestamp", "first"),
                )
        else:
            grouped_df = df.groupby("journey").agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                altitude=("altitude", "max"),
                distance=("distance", "max"),
                journey=("journey", "first"),
                ident = ("ident", "first"),
                model = ("model", "first"),
                timestamp = ("timestamp", "first"),
                )
    
    df_plot = grouped_df.copy()
else:
    df_plot = df.copy()

if st.button('Plot Clusters'):
    if algorithm=="POI":
        plot_point_of_interest(df_plot, altitude_limit, min_samples, R, X, show_unclustered, selected_tags)
    else:
        plot_dbscan(df_plot, altitude_limit, min_samples, eps, algorithm, show_unclustered, selected_tags )

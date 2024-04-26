# Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

# Function to calculate haversine distance
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to calculate distances between points and centroids
def calculate_distances(points, centroids):
    return cdist(points, centroids, metric='haversine')

# Function to filter points based on distance threshold
def filter_points(points, centroids, threshold):
    distances = calculate_distances(points, centroids)
    min_distances = np.min(distances, axis=1)
    return points[min_distances <= threshold]

# Function to find clusters and centroids using DBSCAN
def dbscan_clustering(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    return dbscan.fit(points)

# Function to create convex hulls for clusters
def create_convex_hulls(points, cluster_labels):
    convex_hulls = {}
    for label in np.unique(cluster_labels):
        if label == -1:
            continue
        cluster_points = points[cluster_labels == label]
        hull = ConvexHull(cluster_points)
        convex_hulls[label] = hull
    return convex_hulls

# Function to plot clusters and convex hulls
def plot_clusters(df_clustered, df_unclustered, centroids, convex_hulls):
    # Plot clustered points
    fig = px.scatter_mapbox(df_clustered, lat='latitude', lon='longitude', color='cluster',
                            zoom=10, height=1000, width=1500,
                            title='Drone Detected Position')

    # Plot centroids
    fig.add_trace(go.Scattermapbox(
        lat=centroids[:, 0],
        lon=centroids[:, 1],
        mode='text',
        text=[str(i) for i in range(len(centroids))],
        textfont=dict(size=12, color='white'),
    ))

    # Plot convex hulls
    for label, hull in convex_hulls.items():
        hull_points = np.append(hull.points, [hull.points[0]], axis=0)  # Close the polygon
        fig.add_trace(go.Scattermapbox(
            lat=hull_points[:, 0],
            lon=hull_points[:, 1],
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'Cluster {label}',
            showlegend=False
        ))

    # Plot unclustered points
    fig.add_trace(go.Scattermapbox(
        lat=df_unclustered['latitude'],
        lon=df_unclustered['longitude'],
        mode='markers',
        marker=dict(size=5, color='black'),
        name="Unclustered Points",
    ))

    fig.update_layout(
        mapbox_style="dark",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=True,
        hovermode='closest',
        showlegend=True
    )

    st.plotly_chart(fig)

# Main function to perform clustering and visualization
def main():
    # Load data
    uploaded_files = st.file_uploader("Choose Parquet files", type="parquet", accept_multiple_files=True)
    if not uploaded_files:
        st.warning('Please upload Parquet files.')
        return

    dfs = [pd.read_parquet(file) for file in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)

    # Filter data based on user inputs
    # Add your filtering logic here...

    # Perform clustering
    clustered_points = dbscan_clustering(df[['latitude', 'longitude']].values, eps=eps, min_samples=min_samples)
    df_clustered = df.copy()
    df_clustered['cluster'] = clustered_points.labels_

    # Filter unclustered points
    df_unclustered = df_clustered[df_clustered['cluster'] == -1]
    df_clustered = df_clustered[df_clustered['cluster'] != -1]

    # Calculate centroids and convex hulls
    centroids = df_clustered.groupby('cluster')[['latitude', 'longitude']].mean().values
    convex_hulls = create_convex_hulls(df_clustered[['latitude', 'longitude']].values, df_clustered['cluster'].values)

    # Plot clusters
    plot_clusters(df_clustered, df_unclustered, centroids, convex_hulls)

# Execute the main function
if __name__ == "__main__":
    main()

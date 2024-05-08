import streamlit as st
import pandas as pd
from shapely.geometry import Polygon
import osmnx as ox
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS    
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from io import BytesIO
import googlemaps
import time
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import src.features.info_help as info_help
from src.tools.geomap_tools import haversine, scale_polygon, point_inside_polygon, merge_close_points

class FlightClusterApp:
    def __init__(self):
        self.uploaded_files = None
        self.df_plot = None
        self.df = pd.DataFrame()
        self.grouped_df = pd.DataFrame()
    
    def cluster_dataframe(self):
        # Group by 'cluster' and calculate the necessary statistics
        cluster_df = self.df_alt_filter.groupby('cluster').agg({
            'altitude': ['count', 'max', 'std', 'mean'],
            'distance': ['max', 'std', 'mean'],
            'model': ["nunique", lambda x: x.value_counts().idxmax()],
            'ident': 'nunique',
            'timestamp': lambda x: pd.to_datetime(x).dt.date.nunique()
        })
        
        # Flatten the multi-level index created by groupby
        cluster_df.columns = [' '.join(col).strip() for col in cluster_df.columns.values]
        cluster_df.rename(columns={"altitude count" : "num detections",
                           "model <lambda_0>" : "main model",
                           "timestamp <lambda>" : "num days"}, inplace=True)

        # Round the values to 2 decimal places
        for col in ['altitude max', 'altitude std', 'altitude mean', 
                    'distance max', 'distance std', 'distance mean']:
            cluster_df[col] = cluster_df[col].round(2)

        # Convert the necessary columns to int
        for col in ['num detections', 'ident nunique', 'num days']:
            cluster_df[col] = cluster_df[col].astype(int)

        return cluster_df

    def get_points_of_interest(self):
        lat_min, lat_max = self.df_alt_filter["latitude"].min(), self.df_alt_filter["latitude"].max()
        lon_min, lon_max = self.df_alt_filter["longitude"].min(), self.df_alt_filter["longitude"].max()
        polygon = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
        
        df_points_reduced = pd.DataFrame()
        self.tags_dict = dict()
        if self.api=="OSMNX":
            # Loop through the selected tags
            for tag in self.selected_tags:
                # Split the tag into category and value
                category, value = tag.split(": ")

                # If the category is not in the dictionary, add it with an empty list
                if category not in self.tags_dict:
                    self.tags_dict[category] = []

                # Append the value to the list of values for this category
                self.tags_dict[category].append(value)
                
            df_points_interest = ox.features.features_from_polygon(polygon, tags= self.tags_dict).copy()  # Create a copy to avoid SettingWithCopyWarning
            df_points_interest.dropna(subset=['name'], inplace=True)
            df_points_interest['longitude'] = df_points_interest['geometry'].apply(lambda x: x.centroid.coords.xy[0][0])
            df_points_interest['latitude'] = df_points_interest['geometry'].apply(lambda x: x.centroid.coords.xy[1][0])
            
            for tag in self.tags_dict:
                for elem in self.tags_dict[tag]:
                    cond_i = df_points_interest[tag] == elem
                    points_i = df_points_interest[cond_i][["latitude", "longitude"]].to_numpy()
                    df_temp = pd.DataFrame(merge_close_points(points_i, self.R), columns=["latitude", "longitude"])
                    df_temp["class"] = tag
                    df_temp["type"] = elem
                    df_points_reduced = pd.concat([df_points_reduced, df_temp], ignore_index=True)
            
        elif self.api=="GOOGLE":
            
            max_radius = self.df_alt_filter.apply(lambda row: haversine(polygon.centroid.y, polygon.centroid.x, row['latitude'], row['longitude']), axis=1).max()
            gmaps = googlemaps.Client(key=st.secrets["api_key"]) #self.api_key)
            place_result = gmaps.places_nearby(location=(polygon.centroid.y, polygon.centroid.x), radius=max_radius*1000, keyword='point of interest', language='en')
            time.sleep(2)
            place_result2 = gmaps.places(page_token=place_result['next_page_token'])
            place_result_list = place_result["results"] + place_result2["results"]
            df_points_reduced['latitude'] = [i["geometry"]["location"]["lat"] for i in place_result_list]
            df_points_reduced['longitude'] = [i["geometry"]["location"]["lng"] for i in place_result_list]
            
            for ind, i in enumerate(place_result_list):                 
                class_i = "none"
                type_i = "none"
                try:
                    class_i, type_i = ox.geocoder.geocode_to_gdf(i["name"], which_result=1)[["class", "type"]].values[0]
                except ox._errors.InsufficientResponseError:
                    if len(i["name"].split("-"))>1:
                        try:
                            class_i, type_i = ox.geocoder.geocode_to_gdf(i["name"].split("-")[0], which_result=1)[["class", "type"]].values[0]
                        except ox._errors.InsufficientResponseError:
                            try:
                                class_i, type_i = ox.geocoder.geocode_to_gdf(i["name"].split("-")[1], which_result=1)[["class", "type"]].values[0]
                            except ox._errors.InsufficientResponseError:
                                pass
                df_points_reduced.loc[ind, "class"] = class_i
                df_points_reduced.loc[ind, "type"] = type_i 
                if class_i not in self.tags_dict:
                    self.tags_dict[class_i] = []
                self.tags_dict[class_i].append(type_i)
        
        return df_points_reduced
        
    def plot_point_of_interest(self, df, X, min_samples):        
        # 1st step is to get the points of interest which are already merged in case they are very close, 
        # which is done by giving a R value higher than 0 in the sliders
        centroids = self.df_points_interest[["latitude", "longitude"]].values

        # 2nd step is to get remove those points who are very far (>X) from these centroids
        points = df[["latitude", "longitude"]].to_numpy()
        distances0 = cdist(points, centroids, "euclidean")

        # Find the index of the closest centroid for each point
        clusters0 = np.argmin(distances0, axis=1)

        # Find the minimum distance for each point to its closest centroid and keep those with a closer distance than X
        min_distances = np.min(distances0, axis=1)
        dist_mask = min_distances <= X
        points = points[dist_mask]
        clusters0 = clusters0[dist_mask]
        df_reduced = df[dist_mask].copy()

        # 3rd step is to check now which clusters have a big number of points close (>min_samples)
        # Count the number of points in each cluster
        counts = np.bincount(clusters0)

        # Identify clusters with more than min_samples points
        large_clusters = np.where(counts > min_samples)[0]

        # Filter out small clusters and their centroids
        filtered_centroids = centroids[large_clusters]
        
        # Keep just those POI that accomplish with the previous conditions
        self.df_points_interest = self.df_points_interest.iloc[large_clusters]

        # 4th step is now that some clusters are removed since they didnt have enough points around, some points need will remain in the limbo
        # now, they distances from all the points are calculated to every remaining centroid, those points further than X distance are removed
        # Recalculate the distances
        distances = cdist(points, filtered_centroids, "euclidean")

        # Find the index of the closest centroid for each point
        clusters = np.argmin(distances, axis=1)

        # Find the minimum distance for each point to its closest centroid and keep those with a closer distance than X
        min_distances = np.min(distances, axis=1)
        dist_mask = min_distances <= X
        points = points[dist_mask]
        clusters = clusters[dist_mask]
        df_reduced = df_reduced[dist_mask]
        df_reduced["cluster"] = clusters
        
        df_merged = df.merge(df_reduced[["cluster"]], how='left', left_index=True, right_index=True)
        
        return df_merged['cluster'].fillna(-1).astype(int).values

    def plot_dbscan(self, df, eps, min_samples):  
        
        if self.algorithm=="DBSCAN":
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        elif self.algorithm=="HDBSCAN":
            dbscan = HDBSCAN(cluster_selection_epsilon=eps, min_samples=min_samples)
        elif self.algorithm=="OPTICS":
            dbscan = OPTICS(max_eps=eps, min_samples=min_samples)
            
        # Fit the model to the data
        dbscan.fit(df[['latitude', 'longitude']])
        
        return dbscan.labels_

    def show_graph_table(self):
        # df_alt_filter is the dataframe for all the points with a cluster_id>0
        # df_points_interest is the dataframe that contains all the points of interest

        # Plot the drones points
        df_no_cluster = self.df_alt_filter[self.df_alt_filter["cluster"] == -1]
        df_cluster = self.df_alt_filter[self.df_alt_filter["cluster"] != -1]
        
        silhouette_avg = silhouette_score(df_cluster[['latitude', 'longitude']].values, df_cluster["cluster"])

        clustered_ratio = len(df_cluster) / len(self.df_alt_filter)
        
        st.text(f"Silhouette Avg: {silhouette_avg}")
        st.text(f"Clustered Points Ratio: {np.round(clustered_ratio, 3)}")
        
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
        if self.show_unclustered:
            if not df_no_cluster.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df_no_cluster['latitude'],
                    lon=df_no_cluster['longitude'],
                    mode='markers',  # Include text labels
                    marker=dict(size=5, color='black'),
                    name="",
                ))
        # I dont like this, since if the cluster is too big, the points could not be displayed even being inside the cluster
        # Change it, so it just plots points inside a cluster
        # else:
        #     # Remove the points of interest that are far from the clusters, just for plot visualization only
        #     distances = cdist(self.df_points_interest[["latitude", "longitude"]], centroids_df[["latitude", "longitude"]])
        #     closest_centroid_distances = np.min(distances, axis=1)
        #     max_dist = 0.03
        #     self.df_points_interest = self.df_points_interest[closest_centroid_distances < max_dist]
        
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
            lat=self.df_points_interest['latitude'],
            lon=self.df_points_interest['longitude'],
            mode='markers',  # Include text labels
            marker=dict(size=10, color='white'),
            name="",
        ))
        
        fig.update_layout(
            mapbox_style="dark",
            mapbox_accesstoken=st.secrets["token"], #self.token,
            margin={"r":0,"t":0,"l":0,"b":0},
            autosize=True,
            hovermode='closest',
            showlegend=True
        )
        
        # Compute convex hull for each cluster and plot
        self.df_points_interest['cluster_id'] = -1  # Initialize cluster ID column
        
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
            self.df_points_interest.loc[self.df_points_interest.apply(
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

        df_poi_cluster = self.df_points_interest[self.df_points_interest["cluster_id"] != -1]  
        # create and empty dataframe with an index until the last cluster with a POI inside, the rest will be dismissed.
        if not df_poi_cluster.empty:
            count_df = pd.DataFrame(index=range(df_poi_cluster["cluster_id"].max()+1))
            for tag in self.tags_dict:
                for elem in self.tags_dict[tag]:
                    cond_i = (df_poi_cluster["class"]==tag) & (df_poi_cluster["type"]==elem)
                    count_df[tag + "_" + elem] = df_poi_cluster[cond_i]["cluster_id"].value_counts()
            count_df = count_df.fillna(0)
            count_df.loc['Total', :] = count_df.sum(axis=0)
            count_df["Total"] = count_df.sum(axis=1)
            count_df = count_df[count_df['Total'] != 0] # finally remove those columns which have 0 POI inside
            st.write(count_df)
        else:
            st.text("There are no point of interest inside any cluster")

        cluster_df = self.cluster_dataframe()
        st.write(cluster_df)
        
        self.save_table()
            
        st.plotly_chart(fig)
        
    def save_table(self):    
        
            with BytesIO() as output:
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                self.df_alt_filter.to_excel(writer, sheet_name='Data', index=False)
                parameters_dict = {"altitude_limit" : self.altitude_limit, 
                                "min_samples" : self.min_samples, "centroid_radius" : self.R, 
                                "algorithm" : self.algorithm, "tags" : self.tags_dict}
                
                if self.algorithm=="POI":
                    parameters_dict["min_distance"]=self.X
                else:
                    parameters_dict["max_distance"]=self.eps
                
                df_parameters = pd.DataFrame.from_dict(parameters_dict, orient="index")
                df_parameters.T.to_excel(writer, sheet_name='Parameters', index=False)
                writer.close()
            
                st.download_button(
                    label="Download Excel workbook",
                    data=output.getvalue(),
                    file_name="workbook.xlsx",
                    mime="application/vnd.ms-excel"
                )

    def display_ui(self):
        
        self.upload_and_process_data()
        
        self.api = st.radio("API Selection", ["OSMNX", "GOOGLE"],
            help=info_help.api_help
        )
        
        self.algorithm = st.radio("Cluster Algorithm:", ["DBSCAN", "HDBSCAN", "OPTICS", "POI"], 
                            help = info_help.algorithm_help)
            
        self.altitude_limit = st.slider('Altitude Limit', min_value=0, max_value=500, value=0, step=10,
                           help=info_help.altitude_limit_help)
        
        self.R = st.slider('Centroid Radius', min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%f", 
                  help=info_help.centroid_radius_help)

        if self.algorithm=="POI":
            self.X = st.slider('Min Distance', min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%f",
                        help=info_help.min_dist_help)
        else:
            self.eps = st.slider('Max Distance', min_value=0.001, max_value=0.006, value=0.005, step=0.001, format="%f",
                            help=info_help.max_dist_help)

        self.min_samples = st.slider('Min Samples', min_value=5, max_value=25, value=17,
                        help=info_help.min_sample_help)

        self.selected_tags = st.multiselect('Tags', options=["building: church", "building: chapel", "building: university", "building: hospital", 
                                                    "building: government", "building: stadium", "amenity: prison", "amenity: police"], 
                                default=["building: university"],
                                help=info_help.select_tag_help)
        
        self.show_unclustered = st.checkbox('Show unclustered points', value=False, 
                          help=info_help.show_uncluster_help)


        self.reduce_jrny = st.checkbox('Reduce Journey', value=True, 
                          help=info_help.reduce_jrny_help)

        if self.reduce_jrny:
            mean_max = st.radio(
            "Max or Mean:",
            ["Max", "Mean"],
            help=info_help.max_mean_help) 
            
            grouped_df = pd.DataFrame()
            if not self.df.empty:
                grouped_df = self.df.groupby("journey").agg(
                    latitude=("latitude", "mean"),
                    longitude=("longitude", "mean"),
                    altitude=("altitude", mean_max.lower()),
                    distance=("distance", mean_max.lower()),
                    journey=("journey", "first"),
                    ident = ("ident", "first"),
                    model = ("model", "first"),
                    timestamp = ("timestamp", "first"),
                    )
            
            self.df_plot = grouped_df.copy()
        else:
            self.df_plot = self.df.copy()

    def upload_and_process_data(self):
        self.uploaded_files = st.file_uploader("Choose Parquet files", type="parquet", accept_multiple_files=True)
        dfs = []
        if self.uploaded_files:
            for uploaded_file in self.uploaded_files:
                # Read each parquet file into a DataFrame
                df_i = pd.read_parquet(uploaded_file)
                dfs.append(df_i)

            if dfs:
                self.df = pd.concat(dfs, ignore_index=True)

                # Further processing steps
                self.process_data()

    def process_data(self):
        # Create date and time selectors
        start_time = st.date_input('Start date', value=pd.to_datetime('2023-06-21'))
        end_time =   st.date_input('End date',   value=pd.to_datetime('2023-12-31'))
        
        # Convert the date objects to datetime
        start_time = datetime.combine(start_time, datetime.min.time())
        end_time = datetime.combine(end_time, datetime.min.time())

        # Create a dropdown menu for station names
        station_names = self.df['station_name'].unique().tolist()
        station_names.insert(0, station_names.pop(station_names.index("0QRDKC2R03J32P")))
        selected_station = st.selectbox('Select a station', station_names)

        # Apply the mask
        # Convert 'timestamp' to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Filter 1 based on the user selection
        time_mask = (self.df['timestamp'] > start_time) & (self.df['timestamp'] <= end_time) & (self.df["station_name"] == selected_station)
        self.df = self.df[time_mask]
        
        # Filter 2 to remove those points further than 1.5 times the std
        coordinates_df = self.df[['latitude', 'longitude']]
        z_scores = np.abs((coordinates_df - coordinates_df.mean()) / coordinates_df.std())
        threshold = 2
        mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[mask]
        
        # Haversine distance gives the 2d distance, so the elevation is added to obtain the 3d distance
        self.df['distance'] = self.df.apply(
            lambda df: np.sqrt(
                (haversine(
                    np.radians(df['latitude']), 
                    np.radians(df['longitude']), 
                    np.radians(df['home_lat']), 
                    np.radians(df['home_lon'])
                )**2) + df['elevation']**2
            ), 
            axis=1
        )
        
    def get_score(self, df, fun):
        
        def score_fun(eps, min_samples):
            labels = fun(df, eps, min_samples)
            df["cluster"] = labels
            n_clusters = len(df["cluster"].unique())
            
            if n_clusters <=2 or n_clusters > 50: 
                return 0
            
            df_cluster = df[df["cluster"] != -1]
            clustered_ratio = len(df_cluster) / len(df)
            
            if clustered_ratio < 0.6 or clustered_ratio > 0.95:
                return 0
            
            sil_score = silhouette_score(df_cluster[['latitude', 'longitude']].values, df_cluster["cluster"])
            
            return sil_score * clustered_ratio
        return score_fun
        

    def run(self):
        self.display_ui()
        
        # Process data based on selected algorithm and UI inputs
        if st.button('Plot Clusters'):
            
            self.df_alt_filter = self.df_plot.loc[self.df_plot["altitude"] > self.altitude_limit].copy()
            self.df_alt_filter.reset_index(drop=True, inplace=True)
            
            # 1st step is to merge the points of interest into 1 in case they are very close, which is done by giving a R value higher than 0
            self.df_points_interest = self.get_points_of_interest()
            
            if self.df_plot.empty:
                raise Exception("No files uploaded, please select the desired parquet files")
            else:
                if self.algorithm == "POI":
                    clusters = self.plot_point_of_interest(self.df_alt_filter, self.X, self.min_samples)
                else:
                    clusters = self.plot_dbscan(self.df_alt_filter, self.eps, self.min_samples)
                    
                self.df_alt_filter["cluster"] = clusters
                self.show_graph_table()
                
        if st.button("Optimization"):
            self.df_alt_filter = self.df_plot.loc[self.df_plot["altitude"] > self.altitude_limit].copy()
            self.df_alt_filter.reset_index(drop=True, inplace=True)
            
            score_fun = self.get_score(self.df_alt_filter, self.plot_dbscan)
            
            max_score = 0
            max_score_eps = 0
            max_score_sampl = 0
            for sampl_i in np.arange(5, 25, 1):
                
                for eps_i in np.arange(0.001, 0.006, 0.001):
                    score = score_fun(eps_i, sampl_i)
                    if score > max_score:
                        max_score = score
                        max_score_eps = eps_i
                        max_score_sampl = sampl_i
                        
            st.text(f"Parameters: eps={max_score_eps}, min_sample={max_score_sampl}")
            
            
            
            
# Main code
if __name__ == "__main__":
    app = FlightClusterApp()
    app.run()

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
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import cdist
from io import BytesIO
import googlemaps
import time
from sklearn.metrics import silhouette_score, silhouette_samples
import src.features.info_help as info_help
from src.tools.geomap_tools import haversine, haversine_dist, scale_polygon, point_inside_polygon, merge_close_points
import inspect
# from DBCV import DBCV
import dbcv

class FlightClusterApp:
    def __init__(self):
        self.uploaded_files = None
        self.df_plot = None
        self.df = pd.DataFrame()
        self.grouped_df = pd.DataFrame()
    
    def cluster_dataframe(self) -> pd.DataFrame:
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

    def get_points_of_interest(self) -> pd.DataFrame:
        lat_min, lat_max = self.df_alt_filter["latitude"].min(), self.df_alt_filter["latitude"].max()
        lon_min, lon_max = self.df_alt_filter["longitude"].min(), self.df_alt_filter["longitude"].max()
        polygon = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
        
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
            
            df_points_reduced = pd.DataFrame()
            for tag in self.tags_dict:
                for elem in self.tags_dict[tag]:
                    cond_i = df_points_interest[tag] == elem
                    points_i = df_points_interest[cond_i][["latitude", "longitude"]].to_numpy()
                    df_temp = pd.DataFrame(merge_close_points(points_i, self.R), columns=["latitude", "longitude"])
                    df_temp["class"] = tag
                    df_temp["type"] = elem
                    df_points_reduced = pd.concat([df_points_reduced, df_temp], ignore_index=True)
            self.df_points_interest = df_points_reduced
            
        elif self.api=="GOOGLE":
            if hasattr(self, "polygon") and hasattr(self, "df_points_google"):
                if self.polygon == polygon:
                    self.df_points_interest = self.df_points_google
                    return 0
            else:
                self.polygon = polygon
                    
            maps_places_list = list()
            max_radius = self.df_alt_filter.apply(lambda row: haversine(polygon.centroid.y, polygon.centroid.x, row['latitude'], row['longitude']), axis=1).max()
            gmaps = googlemaps.Client(key=st.secrets["api_key"]) #self.api_key)
            place_result = gmaps.places_nearby(location=(polygon.centroid.y, polygon.centroid.x), radius=max_radius*1000, keyword='point of interest', language='en')
            maps_places_list = place_result["results"]
            while "next_page_token" in place_result.keys():
                time.sleep(2)
                place_result = gmaps.places(page_token=place_result['next_page_token'])
                maps_places_list += place_result["results"]
            
            #maps_places_list = place_result["results"] + place_result2["results"]
            df_points_interest = pd.DataFrame()
            df_points_interest['latitude'] = [i["geometry"]["location"]["lat"] for i in maps_places_list]
            df_points_interest['longitude'] = [i["geometry"]["location"]["lng"] for i in maps_places_list]
            
            for ind, i in enumerate(maps_places_list):                 
                class_i = "none"
                type_i = "none"
                try:
                    class_i, type_i = ox.geocoder.geocode_to_gdf(i["name"], which_result=1)[["class", "type"]].values[0]
                except: #ox._errors.InsufficientResponseError:
                    coordinates_str = str(i["geometry"]["location"]["lat"]) + ", " + str(i["geometry"]["location"]["lng"])
                    ox_response = ox.geocoder.geocode_to_gdf(coordinates_str, which_result=1)
                    class_i, type_i = ox_response[["class", "type"]].values[0]
                    
                df_points_interest.loc[ind, "class"] = class_i
                df_points_interest.loc[ind, "type"] = type_i 
                if class_i not in self.tags_dict:
                    self.tags_dict[class_i] = []
                self.tags_dict[class_i].append(type_i)
                
            self.df_points_google = df_points_interest
            self.df_points_interest = df_points_interest
        
    def plot_point_of_interest(self, df : pd.DataFrame, X : float, min_samples) -> np.ndarray:        
        # 1st step is to get the points of interest which are already merged in case they are very close, 
        # which is done by giving a R value higher than 0 in the sliders
        centroids = self.df_points_interest[["latitude", "longitude"]].values

        # 2nd step is to get remove those points who are very far (>X) from these centroids
        points = df[["latitude", "longitude"]].to_numpy()
        distances0 = cdist(points, centroids, metric=haversine_dist)

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
        distances = cdist(points, filtered_centroids, metric=haversine_dist)

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

    def plot_dbscan(self, df : pd.DataFrame, eps : float, min_samples) -> np.ndarray:  
        
        km_per_radian = 6371.0088
        if self.algorithm=="DBSCAN":
            dbscan = DBSCAN(eps=eps/km_per_radian, min_samples=min_samples, metric="haversine")
        elif self.algorithm=="OPTICS":
            dbscan = OPTICS(max_eps=eps/km_per_radian, min_samples=min_samples, min_cluster_size=max(5, min_samples), metric="haversine")
        elif self.algorithm=="HDBSCAN":
            dbscan = HDBSCAN(cluster_selection_epsilon=eps/km_per_radian, min_samples=min_samples, min_cluster_size=max(5, min_samples), metric="haversine")
            
        # Fit the model to the data
        dbscan.fit(np.radians(df[['latitude', 'longitude']]))
        
        return dbscan.labels_

    def show_graph_table(self):
        # df_alt_filter is the dataframe for all the points with a cluster_id>0
        # df_points_interest is the dataframe that contains all the points of interest
        
        def drop_near_duplicate_coords(df, lat_col="latitude", lon_col="longitude", threshold: float = 1e-9):
            """Drops rows with near-duplicate latitude/longitude values based on Haversine distance."""
            from sklearn.neighbors import NearestNeighbors
            
            if df.shape[0] <= 1:
                return df

            coords = df[[lat_col, lon_col]].values
            nn = NearestNeighbors(n_neighbors=2, metric="haversine")  # Use Haversine distance
            nn.fit(np.radians(coords))  # Convert to radians for haversine calculation
            dists, indices = nn.kneighbors(np.radians(coords), n_neighbors=2)

            # Convert distances from radians to meters
            dists_meters = dists[:, 1] * 6371000  # Earth radius in meters

            # Find near-duplicate rows based on threshold
            to_remove = dists_meters < threshold

            # Return cleaned DataFrame
            return df[~to_remove].reset_index(drop=True)

        # Plot the drones points
        df_no_cluster = self.df_alt_filter[self.df_alt_filter["cluster"] == -1]
        df_cluster = self.df_alt_filter[self.df_alt_filter["cluster"] != -1]
        df_cluster = df_cluster.drop_duplicates(subset=["latitude", "longitude"]) # if not dbcv raises and error, but is this correct?
        df_cluster = drop_near_duplicate_coords(df_cluster, threshold=1e-9)
        
        # Calculate pairwise distances
        distance_matrix = haversine_distances(np.radians(df_cluster[['latitude', 'longitude']]))

        # Use precomputed distance matrix instead of directly using coordinates
        silhouette_avg = silhouette_score(distance_matrix, df_cluster["cluster"], metric='precomputed')
 
        dbcv_score = dbcv.dbcv(df_cluster[['latitude', 'longitude']].values, df_cluster["cluster"].values, precomputed_matrix=distance_matrix, metric="precomputed")
        #dbcv_score = dbcv.dbcv(df_cluster[['latitude', 'longitude']], df_cluster["cluster"].values, metric=haversine_dist)


        clustered_ratio = len(df_cluster) / len(self.df_alt_filter)
        
        st.text(f"DBCV Score: {dbcv_score}")
        st.text(f"Silhouette Avg: {silhouette_avg}")
        st.text(f"Clustered Points Ratio: {np.round(clustered_ratio, 3)}")
        
        color_scale = [
            [0, 'red'],    # Smallest values
            [0.5, 'yellow'], # Middle values
            [1, 'white']   # Largest values
        ]
        
        if "distance_slant_m" in df_cluster:
            fig = px.scatter_mapbox(df_cluster, lat='latitude', lon='longitude', color='distance_slant_m', size="encounters", #color="cluster"
                                    #color_continuous_scale=px.colors.diverging.Portland ,  
                                    color_continuous_scale=color_scale,
                                    #px.colors.cyclical.HSV,
                                    zoom=10, height=1000, width=1500,
                                    title='Drone Detected Position',)
        else:
            fig = px.scatter_mapbox(df_cluster, lat='latitude', lon='longitude', size="altitude", color="cluster",
                        color_continuous_scale=px.colors.diverging.Portland ,  
                        #px.colors.cyclical.HSV,
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
                    marker=dict(size=5, color='yellow'),
                    name="Non-cluster points",
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
            name="Cluster Number",
        ))
        
        #Scatter points of interest
        fig.add_trace(go.Scattermapbox(
            lat=self.df_points_interest['latitude'],
            lon=self.df_points_interest['longitude'],
            mode='markers',  # Include text labels
            marker=dict(size=10, color='green'),
            name="Points of interest",
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
            hull = ConvexHull(points) # it could fail if the data is for pilots, since the data could be really close and the convexhull doesnt work
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
            
            with BytesIO() as output:
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                count_df.to_excel(writer, sheet_name='Data', index=True)
                writer.close()
                st.download_button(
                    label="Download Excel workbook",
                    data=output.getvalue(),
                    file_name="workbook2.xlsx",
                    mime="application/vnd.ms-excel"
                )
                # #Scatter points of interest

        else:
            st.text("There are no point of interest inside any cluster")

        # fig.add_trace(go.Scattermapbox(
        #     lat=df_poi_cluster['latitude'],
        #     lon=df_poi_cluster['longitude'],
        #     mode='markers',  # Include text labels
        #     marker=dict(size=10, color='green'),
        #     name=df_poi_cluster["class"] + " : " + df_poi_cluster["type"],
        # ))

        cluster_df = self.cluster_dataframe()
        st.write(cluster_df)
        
        self.save_table()
            
        st.plotly_chart(fig)
        
    def save_table(self):    
        
            with BytesIO() as output:
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                if self.reduce_jrny:
                    self.df_save = pd.merge(self.df, self.df_alt_filter[["journey", "cluster"]], how="left", on="journey").fillna(-1)
                    self.df_save.to_excel(writer, sheet_name='Data', index=False)
                else:
                    self.df_alt_filter.to_excel(writer, sheet_name='Data', index=False)
                parameters_dict = {"altitude_limit" : self.altitude_limit, 
                                "min_samples" : self.min_samples,
                                "algorithm" : self.algorithm, "tags" : self.tags_dict}
                
                if self.api == "OSMNX":
                    parameters_dict["centroid_radius"]=self.R
                    
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

    def installations_table(self):
        antennas = pd.read_csv("data/external/antennas.csv")
        installations = pd.read_csv("data/external/installations.csv")
        sites = pd.read_csv("data/external/sites.csv")
        antennas.rename(columns={"id" : "antenna_id", "name" : "antenna_name"}, inplace=True)
        sites.rename(columns={"id" : "site_id", "name" : "site_abrev_name", "basename" : "site_name", "code" : "site_code"}, inplace=True)
        installation_merge = pd.merge(installations, antennas, on="antenna_id", how="left")
        self.installation_df = pd.merge(installation_merge, sites, on="site_id", how="left").set_index("id")  
        self.installation_df.start_at = pd.to_datetime(self.installation_df.start_at)
        self.installation_df.end_at = pd.to_datetime(self.installation_df.end_at)

    def display_ui(self):
        print(f"This function {inspect.currentframe().f_code.co_name} was called by {inspect.stack()[1][3]}")
        
        self.uploaded_files = st.file_uploader("Choose Parquet files", type=["parquet", "csv"], accept_multiple_files=True)
        if self.uploaded_files:
            self.upload_and_process_data()
        
        self.api = st.radio("API Selection", ["OSMNX", "GOOGLE"],
            help=info_help.api_help
        )
        
        if self.api=="OSMNX":
            # self.R = st.slider('Centroid Radius', min_value=0.0, max_value=1.0, value=0.5, step=0.02, format="%f", 
            #       help=info_help.centroid_radius_help)
            self.R = 0.5
        
        self.algorithm = st.radio("Cluster Algorithm:", ["DBSCAN", "HDBSCAN", "OPTICS"], 
                           help = info_help.algorithm_help)
        #self.algorithm = "DBSCAN"
            
        self.altitude_limit = st.slider('Altitude Limit', min_value=0, max_value=500, value=5, step=5,
                            help=info_help.altitude_limit_help)

        if self.algorithm=="POI":
            self.X = st.slider('Min Distance', min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%f",
                        help=info_help.min_dist_help)
        else:
            self.eps = st.slider('Max Distance', min_value=0.1, max_value=0.5, value=0.4, step=0.05, format="%f",
                            help=info_help.max_dist_help)

        self.min_samples = st.slider('Min Samples', min_value=3, max_value=12, value=7,
                        help=info_help.min_sample_help)

        if self.api=="OSMNX":
            self.selected_tags = st.multiselect('Tags', options=["building: church", "building: chapel", "building: university", "building: hospital", 
                                                        "building: government", "building: stadium", "amenity: prison", "amenity: police"], 
                                    default=["building: university"],
                                    help=info_help.select_tag_help)
        
        self.show_unclustered = st.checkbox('Show unclustered points', value=False, 
                          help=info_help.show_uncluster_help)

        self.analyze_pilots = st.checkbox('Pilot Analysis', value=False)
        
        self.reduce_jrny = True
        # if self.analyze_pilots:
        #     self.reduce_jrny = True
        # else:
        #     self.reduce_jrny = st.checkbox('Reduce Journey', value=True, 
        #                   help=info_help.reduce_jrny_help)
        
        # Analyze drones or pilots
        if self.analyze_pilots:
            self.df.rename(columns={"home_lat" : "latitude", "home_lon" : "longitude"}, inplace=True)
        else:
            self.df.rename(columns={"drone_latitude" : "latitude", "drone_longitude" : "longitude"}, inplace=True)

        if self.reduce_jrny:
            #mean_max = st.radio("Max or Mean:", ["Max", "Mean"], help=info_help.max_mean_help) 
            #mean_max = "max"
            
            agg_dict = {
                "latitude": ("latitude", "mean"),
                "longitude": ("longitude", "mean"),
                "journey": ("journey", "first"),
                "ident": ("ident", "first"),
                "model": ("model", "first"),
                "timestamp": ("timestamp", "first"),
                "altitude" : ("altitude", "max"),
            }
            
            if "encounters" in self.df.columns:
                agg_dict["encounters"] = ("encounters", "max")
            if "distance" in self.df.columns:
                agg_dict["distance"] = ("distance", "max")
            if "distance_slant_m" in self.df.columns:
                agg_dict["distance_slant_m"] = ("distance_slant_m", "min") # interesed in possible risks when they are very close
            
            grouped_df = pd.DataFrame()
            if not self.df.empty:
                grouped_df = self.df.groupby("journey").agg(**agg_dict)
                
            self.df_plot = grouped_df.copy()
        else:
            self.df_plot = self.df.copy()

    def upload_and_process_data(self):
        print(f"This function {inspect.currentframe().f_code.co_name} was called by {inspect.stack()[1][3]}")
        dfs = []
        for uploaded_file in self.uploaded_files:
            # Read each parquet file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                df_i = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df_i = pd.read_parquet(uploaded_file)
            dfs.append(df_i)

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)

            # Further processing steps
            self.process_data()

    def process_data(self):
        print(f"This function {inspect.currentframe().f_code.co_name} was called by {inspect.stack()[1][3]}")
        
        self.df.rename(columns={"station_name" : "antenna_name"}, inplace=True)
        if "distance_slant_m" in self.df:
            self.df.rename(columns={"drone_lat" : "latitude", 
                                    "drone_lon" : "longitude", 
                                    "drone_height_m" : "altitude", 
                                    "time" : "timestamp", 
                                    "drone_id" : "ident", 
                                    "site" : "site_abrev_name"}, 
                           inplace=True)
            self.df['encounters'] = self.df['en_id'].str.split('-').str[-1].astype(int)
            
        # Apply the mask
        # Convert 'timestamp' to datetime
        if self.df["timestamp"].dtype == object:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        else:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit="s")   
            
        self.df["timestamp"] = self.df["timestamp"].dt.tz_localize(None) 
            
        self.installation_df = self.installation_df[
                                    (self.installation_df.antenna_name.isin(self.df.antenna_name.unique())) &
                                    (self.installation_df.end_at >= self.df.timestamp.min().to_numpy()) &
                                    (self.installation_df.start_at <= self.df.timestamp.max().to_numpy())
                                    ]
        
        # Create a dropdown menu for station names
        station_names = self.installation_df.site_abrev_name.unique().tolist()
        #station_names.insert(0, station_names.pop(station_names.index("0QRDKC2R03J32P")))
        selected_station = st.selectbox('Select a station', station_names)
        
        site_ocurrences = self.installation_df[self.installation_df.site_abrev_name==selected_station]
        
        num_site_ocurrences = site_ocurrences.shape[0]

        if num_site_ocurrences > 1:
            ocurrence_sel = st.selectbox(f"This site has been studied {num_site_ocurrences} times, please select one", list(range(1, num_site_ocurrences + 1)))
        else:
            ocurrence_sel = 1
            
        selected_antenna = site_ocurrences.iloc[ocurrence_sel - 1].antenna_name
        
        default_start_time = site_ocurrences.iloc[ocurrence_sel - 1].start_at
        default_end_time = site_ocurrences.iloc[ocurrence_sel - 1].end_at
        
        # Create date and time selectors
        start_time = st.date_input('Start date', value=pd.to_datetime(default_start_time))
        end_time =   st.date_input('End date',   value=pd.to_datetime(default_end_time))
        
        # Convert the date objects to datetime
        start_time = datetime.combine(start_time, datetime.min.time())
        end_time = datetime.combine(end_time, datetime.min.time())

        # Filter 1 based on the user selection
        time_mask = (self.df['timestamp'] > start_time) & (self.df['timestamp'] <= end_time) & (self.df["antenna_name"] == selected_antenna)
        self.df = self.df[time_mask]
    
        # Rename
        self.df.rename(columns={"latitude": "drone_latitude", "longitude": "drone_longitude"}, inplace=True)
        
        if "station_latitude" in self.df:
            # Filter to remove those points further than radius km from the station
            self.df['station_horiz_distance'] = haversine(self.df['station_latitude'], self.df['station_longitude'], self.df['drone_latitude'], self.df['drone_longitude'])
            
            # Filter the DataFrame to only include rows where the distance is less than or equal to the radius
            radius = 100  # kilometers
            self.df = self.df[self.df['station_horiz_distance'] <= radius]
        
        # Haversine distance gives the 2d distance, so the elevation is added to obtain the 3d distance
        if "distance_home_m" in self.df:
            self.df["distance"] = self.df["distance_home_m"]
        else:
            self.df['distance'] = self.df.apply(
                lambda df: np.sqrt(
                    (haversine(
                        df['drone_latitude'], 
                        df['drone_longitude'], 
                        df['home_lat'], 
                        df['home_lon']
                    )**2) + df['elevation']**2
                ), 
                axis=1
            )
    
    def get_score(self, df: pd.DataFrame, fun):
        
        def score_fun(eps : float, min_samples : int) -> list:
            labels = fun(df, eps, min_samples)
            df["cluster"] = labels
            #n_clusters = len(df["cluster"][df["cluster"]>=0].unique())
            
            n_clusters = df["cluster"].nunique() - 1 # we substract 1 to disregard one cluster associated to the unclustered points
            
            if n_clusters <=1: # or n_clusters>100: 
                return 0, 0, 0, 0, 0
            
            # Test DBCV procedure containing unclustered points:
            df['latitude'] = df['latitude'].round(6)
            df['longitude'] = df['longitude'].round(6)
            df_unclstr = df.drop_duplicates(subset=["latitude", "longitude", "cluster"]).copy()
            distance_matrix = haversine_distances(np.radians(df_unclstr[['latitude', 'longitude']]))
            dbcv_unclstr = dbcv.dbcv(df_unclstr[['latitude', 'longitude']].values, df_unclstr["cluster"].values, precomputed_matrix=distance_matrix, metric="precomputed")
            
            # removing unclustered points for
            df_cluster = df[df["cluster"] != -1]
            cluster_ratio = len(df_cluster) / len(df)
            df_cluster = df_cluster.drop_duplicates(subset=["latitude", "longitude", "cluster"]) # if not dbcv raises an error, but is this correct?
            
            # Calculate pairwise distances in your dataframe
            distance_matrix = haversine_distances(np.radians(df_cluster[['latitude', 'longitude']]))
            # Use precomputed distance matrix instead of directly using coordinates
            silh_score = silhouette_score(distance_matrix, df_cluster["cluster"], metric='precomputed')

            #dbcv_score = dbcv.dbcv(df_cluster[['latitude', 'longitude']].values, df_cluster["cluster"].values, metric=haversine_dist)
            dbcv_score = dbcv.dbcv(df_cluster[['latitude', 'longitude']].values, df_cluster["cluster"].values, precomputed_matrix=distance_matrix, metric="precomputed")
            
            return n_clusters, silh_score, dbcv_score, cluster_ratio, dbcv_unclstr
        
        return score_fun
        
    def run(self):
        print(f"This function {inspect.currentframe().f_code.co_name} was called by {inspect.stack()[1][3]}")
        #self.display_ui()
        
        # Process data based on selected algorithm and UI inputs
        if st.button('Plot Clusters'):
            self.df_alt_filter = self.df_plot.loc[self.df_plot["altitude"] > self.altitude_limit].copy()
            self.df_alt_filter.reset_index(drop=True, inplace=True)
            
            # 1st step is to merge the points of interest into 1 in case they are very close, which is done by giving a R value higher than 0
            self.get_points_of_interest()
            
            if self.df_plot.empty:
                raise Exception("No files uploaded, please select the desired parquet files")
            else:
                if self.algorithm == "POI":
                    clusters = self.plot_point_of_interest(self.df_alt_filter, self.X, self.min_samples)
                else:
                    print(self.df_alt_filter.shape)
                    print(self.eps)
                    clusters = self.plot_dbscan(self.df_alt_filter, self.eps, self.min_samples)
                    
                self.df_alt_filter["cluster"] = clusters
                self.show_graph_table()
                
        if st.button("Optimization"):
            self.df_alt_filter = self.df_plot.loc[self.df_plot["altitude"] > self.altitude_limit].copy()
            self.df_alt_filter.reset_index(drop=True, inplace=True)
            
            score_fun = self.get_score(self.df_alt_filter, self.plot_dbscan)
            
            # Define a function to track the best scores dynamically
            def track_best_scores(scores, score_values, eps_i, sampl_i, n_clusters_i, silh_score_i, dbcv_score_i, cluster_ratio_i, dbcv_unclstr_i):
                for key, value in score_values.items():
                    if value > scores[key]["max_value"]:
                        scores[key]["max_value"] = value
                        scores[key]["eps"] = eps_i
                        scores[key]["sampl"] = sampl_i
                        scores[key]["n_clusters"] = n_clusters_i
                        scores[key]["silh_score"] = silh_score_i
                        scores[key]["dbcv_score"] = dbcv_score_i
                        scores[key]["cluster_ratio"] = cluster_ratio_i
                        scores[key]["dbcv_unclstr"] = dbcv_unclstr_i
                        

            # Initialize dictionaries to track scores
            scores = {
                "dbcv_unclstr": {"max_value": -1,  "eps": 0, "sampl": 0, "n_clusters": 0, "silh_score":0, "dbcv_score": 0, "cluster_ratio":0, "dbcv_unclstr": 0,},
                "f1_silh_score": {"max_value": -1,  "eps": 0, "sampl": 0, "n_clusters": 0, "silh_score":0, "dbcv_score": 0, "cluster_ratio":0, "dbcv_unclstr": 0,},
                "f1_dbcv_score": {"max_value": -1,  "eps": 0, "sampl": 0, "n_clusters": 0, "silh_score":0, "dbcv_score": 0, "cluster_ratio":0, "dbcv_unclstr": 0,},
            }

            # Loop through parameter ranges
            for sampl_i in np.arange(3, 12 + 1, 1):
                for eps_i in np.arange(0.1, 0.6, 0.1):
                    # Get scores from the scoring function
                    
                    n_clusters, silh_score, dbcv_score, cluster_ratio, dbcv_unclstr = score_fun(eps_i, sampl_i)
                    silh_score_rescale = (silh_score + 1) / 2
                    dbcv_score_rescale = (dbcv_score + 1) / 2
                    f1_silh_score = 2 * (silh_score_rescale * cluster_ratio) / (silh_score_rescale + cluster_ratio)
                    f1_dbcv_score = 2 * (dbcv_score_rescale * cluster_ratio) / (dbcv_score_rescale + cluster_ratio)
                    # Track the best scores dynamically
                    track_best_scores(scores, {
                                            #"sil_score": sil_score, "cluster_ratio": cluster_ratio, 
                                            "dbcv_unclstr": dbcv_unclstr, "f1_silh_score": f1_silh_score, "f1_dbcv_score": f1_dbcv_score}, 
                                            eps_i, sampl_i, n_clusters, silh_score_rescale, dbcv_score_rescale, cluster_ratio, dbcv_unclstr)

            test_df = pd.DataFrame.from_dict(scores)
            st.write(test_df.T)
            
# Main code
if __name__ == "__main__":
    app = FlightClusterApp()
    app.installations_table()
    app.display_ui()
    app.run()

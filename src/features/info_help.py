algorithm_help = "Select the algorithm to cluster the flights: \
            DBSCAN: Density-Based Spatial Clustering of Applications with Noise. \
            Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density. \n \
            HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise. Performs DBSCAN over varying epsilon values \
            and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying \
            densities (unlike DBSCAN), and be more robust to parameter selection.\n \
            OPTIC: Ordering Points To Identify the Clustering Structure, closely related to DBSCAN, finds core sample of high density and expands \
            clusters from them [1]. Unlike DBSCAN, keeps cluster hierarchy for a variable neighborhood radius. Better suited for usage on large datasets \
            than the current sklearn implementation of DBSCAN. \n \
            POI: Point of Interest, cluster the points to the nearest point of interest selected by the user"
            
altitude_limit_help = "[Meters] Remove those flights with altitude lower than the limit"           

centroid_radius_help = "[Kilometers] Points of Interest inside the same radius will be averaged into one"

api_help = "API to determine the Points of Interest"

min_dist_help = "[Kilometers] Points further than the minimum distance won't be considered for the cluster creation"

max_dist_help = "[Kilometers] It stands for: The maximum distance between two samples for one to be considered as in the neighborhood of the other. \
                            This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately \
                            for your data set and distance function. Clusters below this value will be merged."
                            
                            
min_sample_help = "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself."


select_tag_help = "Select which Point of Interest are going to be visualized"

show_uncluster_help = "Show those points that haven't been assigned to a cluster"

reduce_jrny_help = "Consider only one point per journey, this point could represent the mean or the maximum values of all the points taken in a journey"

max_mean_help = "The altidude and distance from the pilot considered to create the table will be the mean or the maximum of the journey. Latitude and longitude will always be the mean"

select_files_help = "Select the way of upload the files: \
                    - BROWSER: Upload the files from your computer \
                    - CLICKHOUSE: Upload the files from CLICKHOUSE database"
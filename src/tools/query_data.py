import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from src.tools.airspace_tools import convert_limit
def query_installation_data(client, database):

    antennas_result = client.query(f"SELECT * FROM {database}.antennas")
    antennas_df = pd.DataFrame(antennas_result.result_rows, columns=antennas_result.column_names)

    sites_result = client.query(f"SELECT * FROM {database}.sites")
    sites_df = pd.DataFrame(sites_result.result_rows, columns=sites_result.column_names)

    installations_result = client.query(f"SELECT * FROM {database}.installations")
    installations_df = pd.DataFrame(installations_result.result_rows, columns=installations_result.column_names)
    sites_df.rename(columns={"id" : "site_id", "name" : "site_abrev_name", "basename" : "site_name", "code" : "site_code"}, inplace=True)
    antennas_df.rename(columns={"id" : "antenna_id", "name" : "antenna_name", "type" : "antenna_type"}, inplace=True)
    installations_df.rename(columns={"id" : "installation_id"}, inplace=True)
    data = pd.merge(installations_df, antennas_df, on="antenna_id", how="left")
    final_data = pd.merge(data, sites_df, on="site_id", how="left")
    
    return final_data

def read_airspace_data(folder_path):
    # AIRSPACE DATA from openaip.net
    # Folder containing the JSON files
    folder_path = "/data/pgcasado/projects/ACUTE/data/external/"

    ## List all files ending with .json
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    # Read and concatenate
    dfs = [pd.read_json(os.path.join(folder_path, f)).dropna(axis=1) for f in json_files]
    df = pd.concat(dfs, ignore_index=True)
    #df = pd.read_json("/data/pgcasado/projects/ACUTE/data/external/fr_asp.json").dropna(axis=1)
    df['coordinates'] = df['geometry'].apply(lambda x: np.array(x['coordinates'][0]))
    df["upperLimit"] = df["upperLimit"].apply(convert_limit)
    df["lowerLimit"] = df["lowerLimit"].apply(convert_limit)

    # Convert coordinates to Polygons
    df["geometry"] = df["coordinates"].apply(lambda coords: Polygon(coords))
    
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
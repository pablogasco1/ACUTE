import pandas as pd
import numpy as np
import clickhouse_connect

from shapely.geometry import Point

from src.tools.airspace_tools import get_intersecting_airspace
from src.tools.query_data import query_installation_data, read_airspace_data
from src.tools.airspace_classifier import classify_dataframe

# QUERY DATA
client = clickhouse_connect.get_client(host='192.70.89.35', port=8443, username='acute', password='Odd5Dretkav@', verify=False)
database = client.query("SHOW DATABASES").result_rows[0][0]

query = f"""
    SELECT * 
    FROM acute.pbi_encounters
    WHERE toYear(date) in (2024, 2025)
"""
#     WHERE toDate(date) = '2024-11-16'
#     WHERE toYear(date) in (2024, 2025)

# STEP 1: drone/encounter data
data = client.query(query)
drone_data = pd.DataFrame(data.result_rows, columns=data.column_names)
drone_data.rename(columns={"site" : "site_id"}, inplace=True)
drone_data["geometry"] = drone_data.apply(lambda row: Point(row["drone_lon"], row["drone_lat"]), axis=1)

# STEP 2: installation / antenna data
installation_data = query_installation_data(client, database)

# STEP 3: airspace data
folder_path = "/data/pgcasado/projects/ACUTE/data/external/"
airspace_data = read_airspace_data(folder_path)

# STEP 4: check those airspaces that are within a radius of the installation sites
intersecting_airspaces = get_intersecting_airspace(installation_data, drone_data, airspace_data, radius_km=50)

# STEP 5: add to the drone data the columns with the airspace classification
result_df = classify_dataframe(drone_data, intersecting_airspaces)

result_df.to_parquet("data/drone_airspace_data.parquet")


from shapely.geometry import Point
from pyproj import CRS, Transformer
from shapely.ops import transform
import pandas as pd

def convert_limit(limit_dict):
    value = limit_dict.get('value')
    unit = limit_dict.get('unit')
    if unit==1: # ft
        return value*0.3048 # return in meters
    elif unit==6: # FL
        return value*100*0.3048
    else:
        raise("Unknown unit: {}".format(unit))
    
def classify_airspace(airspace):
    if airspace["type"] == 3 and airspace["icaoClass"] == 8:
        return "PROHIBITED"
    elif airspace["type"] == 1 and airspace["icaoClass"] == 8:
        return "RESTRICTED"
    elif airspace["type"] == 2 and airspace["icaoClass"] == 8:
        return "DANGER"
    elif "CTR" in airspace["name"]:
        return "CTR"
    elif "ATZ" in airspace["name"]:
        return "ATZ"
    elif "CTA" in airspace["name"]:
        return "CTA"
    elif "TMA" in airspace["name"]:
        return "TMA"
    else:
        return "other"
    
    
def get_intersecting_airspace(installation_data, drone_data, airspace_data, radius_km=50):
    
    # Define coordinate reference systems and transformer
    wgs84 = CRS("EPSG:4326")
    utm = CRS(proj="utm", zone=33, ellps="WGS84", south=False)
    transformer_to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(utm, wgs84, always_xy=True)
    
    results = []  # to store intersecting airspaces for each site

    # select those site that appear in our data (drone_data)
    installation_reduced = installation_data[installation_data.site_abrev_name.isin(drone_data.sitename.unique())]

    for _, row in installation_reduced[["site_abrev_name", "latitude", "longitude"]].drop_duplicates().iterrows():

        # Project the point to UTM
        lon, lat = row[["longitude", "latitude"]]
        x, y = transformer_to_utm.transform(lon, lat)
        point_utm = Point(x, y)
        circle_utm = point_utm.buffer(radius_km * 1000)  # meters

        # Convert back to WGS84
        circle_wgs84 = transform(transformer_to_wgs84.transform, circle_utm)

        # Find intersecting airspaces
        intersecting = airspace_data[airspace_data.intersects(circle_wgs84)].copy()

        if not intersecting.empty:
            intersecting["site_abrev_name"] = row["site_abrev_name"]
            intersecting["airspace_type"] = intersecting.apply(classify_airspace, axis=1)
            results.append(intersecting)

    # Combine all sites’ intersections
    return pd.concat(results, ignore_index=True)

    

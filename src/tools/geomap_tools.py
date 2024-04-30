from shapely.geometry import Point, Polygon
import numpy as np

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
from shapely.geometry import Point, Polygon
import numpy as np
from scipy.spatial.distance import pdist, squareform

def point_inside_polygon(point, polygon):
    
    # Create shapely Point and Polygon objects
    shapely_point = Point(point)
    shapely_polygon = Polygon(polygon)
    
    # Check if the point is inside the polygon
    inout = shapely_point.within(shapely_polygon)
    
    return inout

def scale_polygon(polygon, point, factor):
    
    scaled_polygon = []
    for point in polygon:
        scaled_point = point + factor * (point - point)
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

def merge_close_points(points, R):
    # Calculate the pairwise distances between all points
    distances_points = squareform(pdist(points))

    # Initialize a list to hold the new points
    new_points = []
    
    # Initialize a set to keep track of used points
    used_points = set()

    # For each point
    for i in range(len(points)):
        # Skip this point if it has already been used
        if tuple(points[i]) in used_points:
            continue

        # Find the other points that are closer than R distance
        close_points_indices = np.where(distances_points[i] <= R)[0]

        # Mark these points as used
        for index in close_points_indices:
            used_points.add(tuple(points[index]))
        
        close_points = points[close_points_indices]
        mean_point = close_points.mean(axis=0)

        # Add the new point to the list
        new_points.append(mean_point)

    # Convert the list to a numpy array
    points_arr = np.array(new_points)
    
    return points_arr
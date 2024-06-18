from shapely.geometry import Point, Polygon
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import haversine_distances

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

def haversine_dist(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2

    return haversine(lat1, lon1, lat2, lon2)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def merge_close_points(points, R):
    # Calculate the pairwise distances between all points
    # distances_points = squareform(pdist(points, metric=haversine_dist))
    earth_radius = 6371
    distances_points = haversine_distances(np.radians(points)) * earth_radius

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
import pandas as pd
from shapely.geometry import Point
from typing import Dict, Set, Tuple, Any

class AirspaceClassifier:
    """Classifies drone positions against airspace data."""
    
    # Constants
    AIRSPACE_CATEGORIES = ["special_airspace_type", "airspace_type"]
    
    SPECIAL_PRIORITY = {"PROHIBITED": 3, "RESTRICTED": 2, "DANGER": 1}
    CONTROLLED_PRIORITY = {"CTR": 4, "ATZ": 3, "TMA": 2, "CTA": 1}
    ICAO_CLASS_MAPPING = {-1: "", 0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 8: "UNCLASSIFIED"}
    
    def __init__(self, airspace_data: pd.DataFrame):
        """
        Initialize classifier with airspace data.
        
        Args:
            airspace_data: DataFrame containing airspace geometries and metadata
        """
        self.airspace_data = airspace_data
    
    def classify_position(self, point: Point, altitude: float) -> Dict[str, Any]:
        """
        Classify a single position against airspace data.
        
        Args:
            point: Shapely Point (longitude, latitude)
            altitude: Altitude in meters
            
        Returns:
            Dictionary with classification results
        """
        # Find intersecting airspaces
        matches = self._find_intersecting_airspaces(point, altitude)
        
        # Classify intersections
        return self._classify_intersections(matches)
    
    def _find_intersecting_airspaces(self, point: Point, altitude: float) -> pd.DataFrame:
        """Find airspaces that contain the given point at the given altitude."""
        return self.airspace_data[
            self.airspace_data.geometry.contains(point) &
            (self.airspace_data.lowerLimit <= altitude) &
            (self.airspace_data.upperLimit >= altitude)
        ]
    
    def _classify_intersections(self, matches: pd.DataFrame) -> Dict[str, Any]:
        """Classify intersecting airspaces by priority."""
        flags = {cat: "" for cat in self.AIRSPACE_CATEGORIES}
        max_special_priority = 0
        max_controlled_priority = 0
        intersecting_types = set()
        icao_class = -1  # Default: no airspace intersection
        
        for _, airspace in matches.iterrows():
            airspace_type = airspace["airspace_type"]
            
            # Check special airspaces (PROHIBITED, RESTRICTED, DANGER)
            if airspace_type in self.SPECIAL_PRIORITY:
                priority = self.SPECIAL_PRIORITY[airspace_type]
                if priority > max_special_priority:
                    flags["special_airspace_type"] = airspace_type
                    max_special_priority = priority
            
            # Check controlled airspaces (CTR, ATZ, TMA, CTA)
            elif airspace_type in self.CONTROLLED_PRIORITY:
                priority = self.CONTROLLED_PRIORITY[airspace_type]
                if priority > max_controlled_priority:
                    icao_class = airspace["icaoClass"]
                    flags["airspace_type"] = airspace_type
                    max_controlled_priority = priority
            
            # Record intersecting airspace name even if not special or controlled airspace
            intersecting_types.add(airspace["name"])
        
        return {
            "special_airspace_type": flags["special_airspace_type"],
            "airspace_type": flags["airspace_type"],
            "intersecting_airspaces": intersecting_types,
            "icaoClass": self.ICAO_CLASS_MAPPING[icao_class]
        }

def classify_dataframe(df: pd.DataFrame, airspace_data: pd.DataFrame, 
                      lat_col: str = "drone_lat", lon_col: str = "drone_lon", 
                      alt_col: str = "drone_alt_m") -> pd.DataFrame:
    """
    Apply airspace classification to an entire DataFrame.
    
    Args:
        df: DataFrame containing drone positions
        airspace_data: DataFrame containing airspace data
        lat_col: Name of latitude column
        lon_col: Name of longitude column  
        alt_col: Name of altitude column
        
    Returns:
        DataFrame with added classification columns
    """
    # first initialize the classifier with airspace data
    classifier = AirspaceClassifier(airspace_data)
    
    def classify_row(row):
        point = Point(row[lon_col], row[lat_col])
        altitude = row[alt_col]
        return classifier.classify_position(point, altitude)
    
    # Apply classification
    classification_results = df.apply(classify_row, axis=1, result_type='expand')
    classification_results.columns = AirspaceClassifier.AIRSPACE_CATEGORIES + ["intersecting_airspaces", "icaoClass"]
    
    # Merge results with original dataframe
    return pd.concat([df, classification_results], axis=1)
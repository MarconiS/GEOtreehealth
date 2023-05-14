import geopandas as gpd
import networkx as nx
from rtree import index
from shapely.ops import unary_union

# Assuming gdf is your original GeoDataFrame

# First create an R-tree spatial index
def remove_overlapping_polygons(gdf):
    gdf = gdf.reset_index(drop=True)
    idx = index.Index()
    for i, polygon in enumerate(gdf.geometry):
        idx.insert(i, polygon.bounds)

    # Create an undirected graph
    G = nx.Graph()

    # For each polygon
    for i, polygon in gdf.iterrows():
        # Find potential overlapping polygons using R-tree spatial index (much faster than checking all other polygons)
        possible_matches_index = list(idx.intersection(polygon.geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(polygon.geometry)]
        
        # For each overlapping polygon
        for j, overlapping_polygon in precise_matches.iterrows():
            if i != j: # We don't want to compare a polygon to itself
                intersection = polygon.geometry.intersection(overlapping_polygon.geometry)
                if intersection.area > 0.5 * min(polygon.geometry.area, overlapping_polygon.geometry.area):
                    # The intersection is more than 50% of the area of one of the polygons
                    # Add an edge to the graph
                    G.add_edge(i, j)

    # Find connected components in the graph. Each connected component is a group of overlapping polygons that should be merged.
    components = nx.connected_components(G)

    # Merge the polygons in each connected component
    merged_polygons = [unary_union(gdf.geometry.iloc[list(component)]) for component in components]

    # Create a new GeoDataFrame with the merged polygons
    merged_gdf = gpd.GeoDataFrame(geometry=merged_polygons)
    return merged_gdf

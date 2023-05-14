import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from typing import List



def compute_intersection(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the intersection area between two polygons.
    
    Args:
        poly1 (Polygon): The first polygon.
        poly2 (Polygon): The second polygon.
        
    Returns:
        float: Intersection area.
    """
    return poly1.intersection(poly2).area

def compute_union(poly1: Polygon, poly2: Polygon) -> Polygon:
    """
    Compute the union of two polygons.
    
    Args:
        poly1 (Polygon): The first polygon.
        poly2 (Polygon): The second polygon.
        
    Returns:
        Polygon: A new Polygon object representing the union of the input polygons.
    """
    return cascaded_union([poly1, poly2])

def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the Intersection over Union (IoU) between two polygons.
    
    Args:
        poly1 (Polygon): The first polygon.
        poly2 (Polygon): The second polygon.
        
    Returns:
        float: Intersection over Union value.
    """
    intersection_area = compute_intersection(poly1, poly2)
    union_area = poly1.area + poly2.area - intersection_area
    return intersection_area / union_area

def ensemble_polygons(polygons: List[Polygon], iou_threshold: float = 0.5) -> List[Polygon]:
    """
    Ensemble polygons from different object delineation approaches using a greedy approach.
    
    Args:
        polygons (List[Polygon]): List of Polygon objects from different approaches.
        iou_threshold (float): Intersection over Union threshold for ensembling polygons.
        
    Returns:
        List[Polygon]: List of ensembled polygons.
    """
    ensembled_polygons = []

    for poly in polygons:
        matched_polygons = []

        for ensembled_poly in ensembled_polygons:
            iou = compute_iou(poly, ensembled_poly)

            if iou >= iou_threshold:
                matched_polygons.append(ensembled_poly)

        if not matched_polygons:
            ensembled_polygons.append(poly)
        else:
            ensembled_polygons.remove(matched_polygons[0])
            ensembled_polygons.append(compute_union(poly, matched_polygons[0]))

    return ensembled_polygons

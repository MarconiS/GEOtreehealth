import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.geometry import Point, Polygon
from tree_delineation import predict_tree_crowns

def test_predict_tree_crowns():
    # Test 1: Empty batch
    empty_batch = np.array([])
    empty_points = pd.DataFrame(columns=["x", "y", "StemTag"])
    try:
        predict_tree_crowns(empty_batch, empty_points)
    except ValueError as e:
        assert str(e) == "Expected non-empty batch"

    # Test 2: Empty input points
    non_empty_batch = np.random.rand(3, 3, 3)
    try:
        predict_tree_crowns(non_empty_batch, empty_points)
    except ValueError as e:
        assert str(e) == "Expected non-empty input points"

    # Test 3: Different neighbors
    points = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 2, 3],
        "StemTag": [1, 2, 3]
    })
    output = predict_tree_crowns(non_empty_batch, points, neighbors=2)
    assert isinstance(output, tuple)
    
    # Test 4: input_boxes is None and is not None
    input_boxes = np.array([[1, 2, 3, 4]])
    output = predict_tree_crowns(non_empty_batch, points, input_boxes=input_boxes)
    assert isinstance(output, tuple)

    # Test 5: Different point_type values
    for point_type in ['distance', 'cardinal', 'random', 'grid']:
        output = predict_tree_crowns(non_empty_batch, points, point_type=point_type)
        assert isinstance(output, tuple)

    # Test 6: rescale_to is None and not None
    output = predict_tree_crowns(non_empty_batch, points, rescale_to=500)
    assert isinstance(output, tuple)

    # Test 7: mode is 'only_points' and not
    output = predict_tree_crowns(non_empty_batch, points, mode='only_points')
    assert isinstance(output, tuple)

    output = predict_tree_crowns(non_empty_batch, points, mode='not_only_points')  # substitute 'not_only_points' with the actual mode names
    assert isinstance(output, tuple)

    # Test 8: rgb is True and False
    output = predict_tree_crowns(non_empty_batch, points, rgb=True)
    assert isinstance(output, tuple)

    output = predict_tree_crowns(non_empty_batch, points, rgb=False)
    assert isinstance(output, tuple)

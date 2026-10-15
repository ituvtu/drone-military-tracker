import unittest
from Deduplication.geometry_evaluator import GeometryEvaluator

class TestGeometryEvaluator(unittest.TestCase):
    
    def test_empty_boxes(self):
        # Both empty should be redundant (True)
        self.assertTrue(GeometryEvaluator.evaluate_geometry([], []))

    def test_different_lengths(self):
        box = {"label": "tank", "xtl": 10.0, "ytl": 20.0, "xbr": 50.0, "ybr": 60.0}
        self.assertFalse(GeometryEvaluator.evaluate_geometry([box], []))
        self.assertFalse(GeometryEvaluator.evaluate_geometry([], [box]))
        self.assertFalse(GeometryEvaluator.evaluate_geometry([box, box], [box]))

    def test_perfect_match(self):
        box1 = {"label": "tank", "xtl": 10.0, "ytl": 20.0, "xbr": 50.0, "ybr": 60.0}
        box2 = {"label": "tank", "xtl": 10.0, "ytl": 20.0, "xbr": 50.0, "ybr": 60.0}
        self.assertTrue(GeometryEvaluator.evaluate_geometry([box1], [box2]))

    def test_different_labels(self):
        box1 = {"label": "tank", "xtl": 10.0, "ytl": 20.0, "xbr": 50.0, "ybr": 60.0}
        box2 = {"label": "car", "xtl": 10.0, "ytl": 20.0, "xbr": 50.0, "ybr": 60.0}
        self.assertFalse(GeometryEvaluator.evaluate_geometry([box1], [box2]))

    def test_slight_shift_redundant(self):
        # 100x100 box. Shift of 1 pixel horizontally and vertically
        box_ref = {"label": "tank", "xtl": 100.0, "ytl": 100.0, "xbr": 200.0, "ybr": 200.0}
        box_curr = {"label": "tank", "xtl": 101.0, "ytl": 101.0, "xbr": 201.0, "ybr": 201.0}
        
        # Verify it meets all criteria
        # Area = 10000.
        # Intersection = 99 * 99 = 9801. Union = 20000 - 9801 = 10199. IoU = 9801/10199 = 0.9610 > 0.95
        # d_c = sqrt(2) / 1468.60 = 0.00096 < 0.01
        # s_e = 1.0 (between 0.98 and 1.02)
        self.assertTrue(GeometryEvaluator.evaluate_geometry([box_ref], [box_curr]))

    def test_large_shift_not_redundant(self):
        # 100x100 box. Shift of 5 pixels (IoU will be 90.25 / 109.75 = 0.822 < 0.95)
        box_ref = {"label": "tank", "xtl": 100.0, "ytl": 100.0, "xbr": 200.0, "ybr": 200.0}
        box_curr = {"label": "tank", "xtl": 105.0, "ytl": 105.0, "xbr": 205.0, "ybr": 205.0}
        self.assertFalse(GeometryEvaluator.evaluate_geometry([box_ref], [box_curr]))

    def test_slight_size_change_redundant(self):
        # Reference: 100x100. Current: 100x101 (centered vertically)
        box_ref = {"label": "tank", "xtl": 100.0, "ytl": 100.0, "xbr": 200.0, "ybr": 200.0}
        box_curr = {"label": "tank", "xtl": 100.0, "ytl": 99.5, "xbr": 200.0, "ybr": 200.5}
        
        # IoU = 10000 / 10100 = 0.990 > 0.95
        # d_c = 0 (perfect center match)
        # s_e = 10000 / 10100 = 0.990 (between 0.98 and 1.02)
        self.assertTrue(GeometryEvaluator.evaluate_geometry([box_ref], [box_curr]))

    def test_large_size_change_not_redundant(self):
        # Area changes from 10000 to 10500. s_e = 10000/10500 = 0.952 < 0.98
        box_ref = {"label": "tank", "xtl": 100.0, "ytl": 100.0, "xbr": 200.0, "ybr": 200.0}
        box_curr = {"label": "tank", "xtl": 100.0, "ytl": 97.5, "xbr": 200.0, "ybr": 202.5}
        self.assertFalse(GeometryEvaluator.evaluate_geometry([box_ref], [box_curr]))

    def test_multiple_boxes_matching(self):
        # Two boxes: tank and car
        ref = [
            {"label": "tank", "xtl": 10.0, "ytl": 10.0, "xbr": 50.0, "ybr": 50.0},
            {"label": "car", "xtl": 100.0, "ytl": 100.0, "xbr": 150.0, "ybr": 150.0}
        ]
        # Same boxes but shuffled order and slightly shifted
        curr = [
            {"label": "car", "xtl": 100.5, "ytl": 100.5, "xbr": 150.5, "ybr": 150.5},
            {"label": "tank", "xtl": 10.1, "ytl": 10.1, "xbr": 50.1, "ybr": 50.1}
        ]
        self.assertTrue(GeometryEvaluator.evaluate_geometry(ref, curr))

    def test_multiple_boxes_one_fails(self):
        ref = [
            {"label": "tank", "xtl": 10.0, "ytl": 10.0, "xbr": 50.0, "ybr": 50.0},
            {"label": "car", "xtl": 100.0, "ytl": 100.0, "xbr": 150.0, "ybr": 150.0}
        ]
        # Car matches, but tank is shifted too far
        curr = [
            {"label": "car", "xtl": 100.5, "ytl": 100.5, "xbr": 150.5, "ybr": 150.5},
            {"label": "tank", "xtl": 20.0, "ytl": 20.0, "xbr": 60.0, "ybr": 60.0}
        ]
        self.assertFalse(GeometryEvaluator.evaluate_geometry(ref, curr))

if __name__ == "__main__":
    unittest.main()

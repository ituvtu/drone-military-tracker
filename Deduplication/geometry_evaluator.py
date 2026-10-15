import numpy as np
from typing import List, Dict, Any

class GeometryEvaluator:
    """Evaluates spatial staticity and redundancy of moving objects between frames."""
    
    @staticmethod
    def evaluate_geometry(
        bboxes_ref: List[Dict[str, Any]],
        bboxes_curr: List[Dict[str, Any]],
        img_width: float = 1280.0,
        img_height: float = 720.0
    ) -> bool:
        """
        Calculates IoU, normalized centroid distance, and area change ratio for objects
        between a reference frame and a current frame.
        
        Objects are matched greedily between the lists.
        
        Args:
            bboxes_ref: List of reference bounding boxes.
            bboxes_curr: List of current bounding boxes.
            img_width: Width of the frames.
            img_height: Height of the frames.
            
        Returns:
            bool: True if the frames are geometrically redundant (all objects are static), else False.
        """
        if len(bboxes_ref) != len(bboxes_curr):
            return False
            
        if not bboxes_ref:
            # Both are empty lists, technically redundant/identical (background frames)
            return True
            
        # Copy to track matched boxes
        remaining_curr = list(bboxes_curr)
        
        # Calculate image diagonal
        diag = np.sqrt(img_width**2 + img_height**2)
        if diag <= 0:
            diag = 1.0 # Prevent division by zero
            
        for box_ref in bboxes_ref:
            label_ref = box_ref.get("label")
            xtl_a, ytl_a = box_ref["xtl"], box_ref["ytl"]
            xbr_a, ybr_a = box_ref["xbr"], box_ref["ybr"]
            
            area_a = (xbr_a - xtl_a) * (ybr_a - ytl_a)
            xc_a = (xtl_a + xbr_a) / 2.0
            yc_a = (ytl_a + ybr_a) / 2.0
            
            matched_idx = -1
            for idx, box_curr in enumerate(remaining_curr):
                if box_curr.get("label") != label_ref:
                    continue
                    
                xtl_b, ytl_b = box_curr["xtl"], box_curr["ytl"]
                xbr_b, ybr_b = box_curr["xbr"], box_curr["ybr"]
                
                area_b = (xbr_b - xtl_b) * (ybr_b - ytl_b)
                if area_a <= 0 or area_b <= 0:
                    continue
                    
                # 1. Intersection over Union (IoU)
                int_xtl = max(xtl_a, xtl_b)
                int_ytl = max(ytl_a, ytl_b)
                int_xbr = min(xbr_a, xbr_b)
                int_ybr = min(ybr_a, ybr_b)
                
                int_w = max(0.0, int_xbr - int_xtl)
                int_h = max(0.0, int_ybr - int_ytl)
                area_int = int_w * int_h
                
                area_union = area_a + area_b - area_int
                if area_union <= 0:
                    continue
                    
                iou = area_int / area_union
                
                # 2. Normalized Centroid Distance (d_c)
                xc_b = (xtl_b + xbr_b) / 2.0
                yc_b = (ytl_b + ybr_b) / 2.0
                dist = np.sqrt((xc_a - xc_b)**2 + (yc_a - yc_b)**2)
                dc = dist / diag
                
                # 3. Area Change Ratio (s_e)
                se = area_a / area_b
                
                # Check static thresholds: IoU > 0.95, dc < 0.01, 0.98 <= se <= 1.02
                if iou > 0.95 and dc < 0.01 and 0.98 <= se <= 1.02:
                    matched_idx = idx
                    break
                    
            if matched_idx == -1:
                return False
                
            remaining_curr.pop(matched_idx)
            
        return True

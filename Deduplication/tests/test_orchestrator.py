import unittest
import tempfile
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

from Deduplication.orchestrator import DeduplicationOrchestrator
from Deduplication.frame_loader import FrameLoader

class TestDeduplicationOrchestrator(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        
        self.source_dir = self.base_path / "dataset_v12"
        self.target_dir = self.base_path / "dataset_v13"
        self.source_dir.mkdir()
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def create_annotated_image(self, path: Path, shape: tuple, draw_rects: list, blur: bool = False):
        # Create image
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for pt1, pt2 in draw_rects:
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1)
            
        if blur:
            img = cv2.GaussianBlur(img, (9, 9), 0)
            
        cv2.imwrite(str(path), img)

    def test_complete_migration_flow(self):
        # Set up source episodes
        ep1_dir = self.source_dir / "map_1" / "ds_1" / "ep_001"
        ep2_dir = self.source_dir / "map_1" / "ds_1" / "ep_002_neg"
        ep3_dir = self.source_dir / "map_1" / "ds_1" / "ep_003"
        
        for d in [ep1_dir, ep2_dir, ep3_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # 1. ep_001: Has three frames
        # f_001: Sharp, white box at (10, 10)
        self.create_annotated_image(ep1_dir / "f_001.jpg", (100, 100), [((10, 10), (40, 40))], blur=False)
        # f_002: Blurry, duplicate of f_001
        self.create_annotated_image(ep1_dir / "f_002.jpg", (100, 100), [((10, 10), (40, 40))], blur=True)
        # f_003: Sharp, unique, white box at (60, 60)
        self.create_annotated_image(ep1_dir / "f_003.jpg", (100, 100), [((60, 60), (90, 90))], blur=False)
        
        xml_ep1 = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <version>1.1</version>
          <meta>
            <task><id>1</id><name>ep_001</name></task>
          </meta>
          <track id="0" label="tank">
            <box frame="0" outside="0" xtl="10.0" ytl="10.0" xbr="40.0" ybr="40.0" />
            <box frame="1" outside="0" xtl="10.0" ytl="10.0" xbr="40.0" ybr="40.0" />
          </track>
          <track id="1" label="car">
            <box frame="2" outside="0" xtl="60.0" ytl="60.0" xbr="90.0" ybr="90.0" />
          </track>
        </annotations>
        """
        (ep1_dir / "annotations.xml").write_text(xml_ep1, encoding="utf-8")
        
        # 2. ep_002_neg: Has 1 background frame (all zeros)
        self.create_annotated_image(ep2_dir / "f_001.jpg", (100, 100), [], blur=False)
        
        # 3. ep_003: Has 1 frame that is a duplicate of ep_001/f_001 but even sharper
        # We make it even sharper by drawing multiple overlapping boxes or a high-contrast pattern
        self.create_annotated_image(ep3_dir / "f_001.jpg", (100, 100), [((10, 10), (40, 40))], blur=False)
        
        xml_ep3 = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <version>1.1</version>
          <meta>
            <task><id>3</id><name>ep_003</name></task>
          </meta>
          <track id="0" label="tank">
            <box frame="0" outside="0" xtl="10.0" ytl="10.0" xbr="40.0" ybr="40.0" />
          </track>
        </annotations>
        """
        (ep3_dir / "annotations.xml").write_text(xml_ep3, encoding="utf-8")
        
        # Run process migration
        config = {
            "beta": 1.0,
            "dhash_threshold": 8,
            "img_width": 100.0,
            "img_height": 100.0
        }
        success = DeduplicationOrchestrator.process_balanced_migration(
            str(self.source_dir), str(self.target_dir), config
        )
        
        self.assertTrue(success)
        
        # Verify target structure
        target_ep1 = self.target_dir / "map_1" / "ds_1" / "ep_001"
        target_ep2 = self.target_dir / "map_1" / "ds_1" / "ep_002_neg"
        target_ep3 = self.target_dir / "map_1" / "ds_1" / "ep_003"
        
        self.assertTrue(target_ep1.exists())
        self.assertTrue(target_ep2.exists())
        self.assertTrue(target_ep3.exists())
        
        # Verify ep_002_neg background frame exists
        self.assertTrue((target_ep2 / "f_001.jpg").exists())
        
        # Verify duplicate replacement and deletion:
        # - ep_001/f_002.jpg was a blurry duplicate, so it was skipped. It must NOT exist in target_ep1.
        self.assertFalse((target_ep1 / "f_002.jpg").exists())
        
        # - ep_001/f_003.jpg was unique, so it must exist in target_ep1.
        self.assertTrue((target_ep1 / "f_003.jpg").exists())
        
        # - ep_003/f_001.jpg replaces ep_001/f_001.jpg because score_new > score_stored.
        # So target_ep1/f_001.jpg must NOT exist, and target_ep3/f_001.jpg MUST exist.
        self.assertFalse((target_ep1 / "f_001.jpg").exists())
        self.assertTrue((target_ep3 / "f_001.jpg").exists())
        
        # Verify XML finalization and frame index shifting in ep_001
        # ep_001 has only f_003.jpg remaining, which is frame 0 in target list.
        xml_file1 = target_ep1 / "annotations.xml"
        self.assertTrue(xml_file1.exists())
        tree1 = ET.parse(xml_file1)
        root1 = tree1.getroot()
        
        # Car track box frame should be shifted from 2 to 0
        boxes = root1.findall(".//track[@label='car']/box")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].get("frame"), "0")
        
        # Tank track should have no boxes or not exist (or exist with empty boxes)
        tank_boxes = root1.findall(".//track[@label='tank']/box")
        self.assertEqual(len(tank_boxes), 0)
        
        # Verify XML finalization in ep_003
        xml_file3 = target_ep3 / "annotations.xml"
        self.assertTrue(xml_file3.exists())
        tree3 = ET.parse(xml_file3)
        root3 = tree3.getroot()
        
        # Tank track box frame should be 0
        tank_boxes3 = root3.findall(".//track[@label='tank']/box")
        self.assertEqual(len(tank_boxes3), 1)
        self.assertEqual(tank_boxes3[0].get("frame"), "0")
        
        # Verify log file generated
        log_file = self.target_dir / "dataset_actions.log"
        self.assertTrue(log_file.exists())
        log_content = log_file.read_text(encoding="utf-8")
        self.assertIn("Starting migration", log_content)
        self.assertIn("REPLACE: map_1/ds_1/ep_003/f_001.jpg", log_content)
        self.assertIn("SKIP: Duplicate", log_content)
        self.assertIn("Migration complete!", log_content)

if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from Deduplication.frame_loader import Frame, FrameLoader, frame_idx_from_name

class TestFrameLoader(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def create_dummy_image(self, path: Path):
        # Create a 10x10 dummy BGR image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        
    def test_frame_idx_from_name(self):
        self.assertEqual(frame_idx_from_name("f_011.jpg"), 11)
        self.assertEqual(frame_idx_from_name("frame_023.png"), 23)
        self.assertEqual(frame_idx_from_name("image_9.jpeg"), 9)
        self.assertEqual(frame_idx_from_name("123.jpg"), 123)
        with self.assertRaises(ValueError):
            frame_idx_from_name("no_digits.jpg")
            
    def test_frame_load_image(self):
        img_path = self.base_path / "f_001.jpg"
        self.create_dummy_image(img_path)
        
        frame = Frame(img_path, 1, [], True)
        img = frame.load_image()
        
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (10, 10, 3))
        
        # Test missing image throws IOError
        frame_bad = Frame(self.base_path / "non_existent.jpg", 2, [], True)
        with self.assertRaises(IOError):
            frame_bad.load_image()

    def test_parse_track_based_xml(self):
        image_files = [
            Path("f_001.jpg"), # Index 0 in CVAT maps to f_001 (frame_idx = 1)
            Path("f_002.jpg"), # Index 1 in CVAT maps to f_002 (frame_idx = 2)
            Path("f_003.jpg"), # Index 2 in CVAT maps to f_003 (frame_idx = 3)
        ]
        
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <version>1.1</version>
          <meta>
            <task><id>123</id></task>
          </meta>
          <track id="0" label="tank">
            <box frame="0" outside="0" xtl="10.0" ytl="20.0" xbr="50.0" ybr="60.0" />
            <box frame="1" outside="0" xtl="15.0" ytl="25.0" xbr="55.0" ybr="65.0" />
            <box frame="2" outside="1" xtl="10.0" ytl="20.0" xbr="50.0" ybr="60.0" />
          </track>
          <track id="1" label="car">
            <box frame="1" outside="0" xtl="100.0" ytl="100.0" xbr="150.0" ybr="150.0" />
            <!-- Degenerate box -->
            <box frame="0" outside="0" xtl="100.0" ytl="100.0" xbr="90.0" ybr="90.0" />
          </track>
        </annotations>
        """
        xml_path = self.base_path / "annotations.xml"
        xml_path.write_text(xml_content, encoding="utf-8")
        
        annotations, meta, fmt_type = FrameLoader.parse_cvat_xml(xml_path, image_files)
        
        self.assertEqual(fmt_type, "track")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.find("task/id").text, "123")
        
        # CVAT frame 0 -> maps to f_001.jpg (real index 1)
        self.assertIn(1, annotations)
        self.assertEqual(len(annotations[1]), 1)
        self.assertEqual(annotations[1][0]["label"], "tank")
        
        # CVAT frame 1 -> maps to f_002.jpg (real index 2)
        self.assertIn(2, annotations)
        self.assertEqual(len(annotations[2]), 2)
        
        # CVAT frame 2 -> outside=1 -> should map and have outside="1"
        self.assertIn(3, annotations)
        self.assertEqual(annotations[3][0]["outside"], "1")

    def test_parse_image_based_xml(self):
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <version>1.1</version>
          <meta>
            <task><id>456</id></task>
          </meta>
          <image id="0" name="f_000.jpg">
            <box label="tank" xtl="10.0" ytl="20.0" xbr="50.0" ybr="60.0" />
          </image>
          <image id="1" name="f_001.jpg">
            <box label="car" xtl="100.0" ytl="100.0" xbr="150.0" ybr="150.0" />
          </image>
        </annotations>
        """
        xml_path = self.base_path / "annotations.xml"
        xml_path.write_text(xml_content, encoding="utf-8")
        
        annotations, meta, fmt_type = FrameLoader.parse_cvat_xml(xml_path, [])
        
        self.assertEqual(fmt_type, "image")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.find("task/id").text, "456")
        
        self.assertIn(0, annotations)
        self.assertEqual(annotations[0][0]["label"], "tank")
        
        self.assertIn(1, annotations)
        self.assertEqual(annotations[1][0]["label"], "car")

    def test_load_episode_negative(self):
        ep_dir = self.base_path / "ep_001_neg"
        ep_dir.mkdir()
        
        self.create_dummy_image(ep_dir / "f_002.jpg")
        self.create_dummy_image(ep_dir / "f_001.jpg")
        
        frames, meta, fmt_type = FrameLoader.load_episode(str(ep_dir))
        
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].frame_idx, 1)
        self.assertEqual(frames[1].frame_idx, 2)
        self.assertIsNone(meta)
        self.assertIsNone(fmt_type)

    def test_load_episode_positive(self):
        ep_dir = self.base_path / "ep_001"
        ep_dir.mkdir()
        
        self.create_dummy_image(ep_dir / "f_001.jpg")
        self.create_dummy_image(ep_dir / "f_002.jpg")
        
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
          <meta>
            <task><id>789</id></task>
          </meta>
          <track id="0" label="tank">
            <box frame="0" outside="0" xtl="10.0" ytl="20.0" xbr="50.0" ybr="60.0" />
          </track>
        </annotations>
        """
        (ep_dir / "annotations.xml").write_text(xml_content, encoding="utf-8")
        
        frames, meta, fmt_type = FrameLoader.load_episode(str(ep_dir))
        
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].frame_idx, 1)
        self.assertEqual(frames[1].frame_idx, 2)
        self.assertEqual(fmt_type, "track")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.find("task/id").text, "789")
        
        self.assertFalse(frames[0].is_background)
        self.assertEqual(len(frames[0].bboxes), 1)
        self.assertTrue(frames[1].is_background)
        self.assertEqual(len(frames[1].bboxes), 0)

if __name__ == "__main__":
    unittest.main()

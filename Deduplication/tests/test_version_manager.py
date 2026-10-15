import unittest
import tempfile
import shutil
from pathlib import Path
from Deduplication.version_manager import DatasetVersionManager

class TestDatasetVersionManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        
    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()
        
    def test_invalid_source_dir(self):
        # Test that ValueError is raised for non-existent source directory
        with self.assertRaises(ValueError):
            DatasetVersionManager.initialize_next_version(str(self.base_path / "non_existent"))
            
        # Test that ValueError is raised if source path is not a directory
        file_path = self.base_path / "file.txt"
        file_path.touch()
        with self.assertRaises(ValueError):
            DatasetVersionManager.initialize_next_version(str(file_path))

    def test_auto_increment_version(self):
        # Test version name auto-increment logic
        source_dir = self.base_path / "dataset_v12"
        source_dir.mkdir()
        
        # We need map/ds/ep inside to see replication, but first let's just check the returned target path
        target_path_str = DatasetVersionManager.initialize_next_version(str(source_dir))
        target_path = Path(target_path_str)
        
        self.assertEqual(target_path.name, "dataset_v13")
        self.assertTrue(target_path.exists())
        self.assertTrue(target_path.is_dir())
        
        # Test custom prefix with version
        source_dir_2 = self.base_path / "my_custom_dataset_v01"
        source_dir_2.mkdir()
        target_path_str_2 = DatasetVersionManager.initialize_next_version(str(source_dir_2))
        self.assertEqual(Path(target_path_str_2).name, "my_custom_dataset_v2")

        # Test non-versioned name fallback
        source_dir_3 = self.base_path / "dataset_no_version"
        source_dir_3.mkdir()
        target_path_str_3 = DatasetVersionManager.initialize_next_version(str(source_dir_3))
        self.assertEqual(Path(target_path_str_3).name, "dataset_no_version_v13")

    def test_custom_target_dir(self):
        source_dir = self.base_path / "dataset_v12"
        source_dir.mkdir()
        custom_target = self.base_path / "custom_target_folder"
        
        target_path_str = DatasetVersionManager.initialize_next_version(
            str(source_dir), target_dir=str(custom_target)
        )
        
        self.assertEqual(target_path_str, str(custom_target.resolve()))
        self.assertTrue(custom_target.exists())
        self.assertTrue(custom_target.is_dir())

    def test_structure_replication(self):
        # Create map_X/ds_Y/ep_Z structure
        source_dir = self.base_path / "dataset_v12"
        
        # Good paths
        path1 = source_dir / "map_1" / "ds_1" / "ep_001"
        path2 = source_dir / "map_1" / "ds_2" / "ep_002_neg"
        path3 = source_dir / "map_2" / "ds_1" / "ep_003"
        
        # Bad paths (should not be replicated)
        bad_path1 = source_dir / "map_1" / "ds_1" / "not_an_episode"
        bad_path2 = source_dir / "other_folder" / "ds_1" / "ep_001"
        bad_path3 = source_dir / "map_1" / "other_folder" / "ep_001"
        
        for path in [path1, path2, path3, bad_path1, bad_path2, bad_path3]:
            path.mkdir(parents=True, exist_ok=True)
            # Add some dummy files to mock content
            (path / "f_001.jpg").touch()
            (path / "annotations.xml").touch()
            
        # Run replication
        target_path_str = DatasetVersionManager.initialize_next_version(str(source_dir))
        target_path = Path(target_path_str)
        
        # Verify valid paths exist and are empty
        for rel_path in [
            Path("map_1") / "ds_1" / "ep_001",
            Path("map_1") / "ds_2" / "ep_002_neg",
            Path("map_2") / "ds_1" / "ep_003"
        ]:
            recreated = target_path / rel_path
            self.assertTrue(recreated.exists(), f"Path should be recreated: {rel_path}")
            self.assertTrue(recreated.is_dir())
            # Should be empty
            self.assertEqual(len(list(recreated.iterdir())), 0, f"Path should be empty: {rel_path}")
            
        # Verify invalid paths DO NOT exist in target
        for rel_path in [
            Path("map_1") / "ds_1" / "not_an_episode",
            Path("other_folder") / "ds_1" / "ep_001",
            Path("map_1") / "other_folder" / "ep_001"
        ]:
            not_recreated = target_path / rel_path
            self.assertFalse(not_recreated.exists(), f"Path should NOT be recreated: {rel_path}")

if __name__ == "__main__":
    unittest.main()

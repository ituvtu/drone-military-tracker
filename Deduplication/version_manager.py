import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class DatasetVersionManager:
    """Manages dataset versions and directory mirroring."""
    
    @staticmethod
    def initialize_next_version(source_dir: str, target_dir: Optional[str] = None) -> str:
        """
        Scans source_dir for map_X/ds_Y/ep_Z directory structures and creates
        an identical empty directory structure in target_dir.
        
        If target_dir is not specified, it is automatically derived by incrementing
        the version number of the source_dir (e.g., dataset_v12 -> dataset_v13).
        
        Args:
            source_dir: Absolute path to the source dataset.
            target_dir: Optional absolute path to the target dataset directory.
            
        Returns:
            The absolute path of the created target directory.
            
        Raises:
            ValueError: If source_dir does not exist or is not a directory.
        """
        source_path = Path(source_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Source path is not a directory: {source_dir}")
            
        if target_dir is None:
            # Auto-increment version
            parent_dir = source_path.parent
            source_name = source_path.name
            
            # Look for version pattern like dataset_v12, v12, etc.
            match = re.search(r'(.*?_v|v)(\d+)$', source_name, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                version = int(match.group(2))
                target_name = f"{prefix}{version + 1}"
            else:
                target_name = f"{source_name}_v13"
            target_path = parent_dir / target_name
        else:
            target_path = Path(target_dir).resolve()
            
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Walk and recreate map_X/ds_Y/ep_Z
        for root, dirs, files in os.walk(source_path):
            root_path = Path(root)
            try:
                rel_path = root_path.relative_to(source_path)
            except ValueError:
                continue
                
            parts = rel_path.parts
            # Check if the path matches the map_X/ds_Y/ep_Z structure
            if len(parts) == 3:
                is_map = parts[0].startswith("map_")
                is_ds = parts[1].startswith("ds_")
                is_ep = parts[2].startswith("ep_")
                if is_map and is_ds and is_ep:
                    # Recreate this folder in target_path
                    dest_dir = target_path / rel_path
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Recreated directory: {rel_path}")
                    
        return str(target_path)

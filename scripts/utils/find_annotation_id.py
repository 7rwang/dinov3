#!/usr/bin/env python3
"""
Script to find a specific desc_id across all scene description JSON files.
Usage: python find_annotation_id.py <target_desc_id>
"""

import json
import os
import sys
from pathlib import Path


def find_desc_id(data_dir, target_id):
    """
    Search for a target desc_id across all scene description files.
    
    Args:
        data_dir: Path to the data directory containing scene folders
        target_id: The desc_id to search for
    
    Returns:
        List of dictionaries containing matches with scene info
    """
    matches = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return matches
    
    # Get all scene directories
    scene_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Searching for desc_id '{target_id}' in {len(scene_dirs)} scenes...")
    
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        description_file = scene_dir / f"{scene_id}_descriptions.json"
        
        if not description_file.exists():
            continue
            
        try:
            with open(description_file, 'r') as f:
                data = json.load(f)
                
            # Search through descriptions
            if 'descriptions' in data:
                for i, description in enumerate(data['descriptions']):
                    if description.get('desc_id') == target_id:
                        match_info = {
                            'scene_id': scene_id,
                            'file_path': str(description_file),
                            'description_index': i,
                            'description': description
                        }
                        matches.append(match_info)
                        print(f"Found in scene {scene_id} at index {i}")
                        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {description_file}: {e}")
            continue
    
    return matches


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_annotation_id.py <target_desc_id>")
        print("Example: python find_annotation_id.py b0f35632-a0c6-4e52-ad97-36a7b5bdf65d")
        sys.exit(1)
    
    target_id = sys.argv[1]
    data_dir = "/nas/qirui/scenefun3d/data"
    
    matches = find_desc_id(data_dir, target_id)
    
    if matches:
        print(f"\nFound {len(matches)} match(es) for desc_id '{target_id}':")
        for match in matches:
            print(f"\nScene ID: {match['scene_id']}")
            print(f"File: {match['file_path']}")
            print(f"Description index: {match['description_index']}")
            print(f"Description: {match['description']['description']}")
            print(f"Associated annot_id(s): {match['description']['annot_id']}")
    else:
        print(f"No matches found for desc_id '{target_id}'")


if __name__ == "__main__":
    main()
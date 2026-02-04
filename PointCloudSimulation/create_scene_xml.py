import os
from pathlib import Path


def create_scene_xml(bridge, output_path):
    """Create scene XML file for a bridge with all its components.
    
    Args:
        bridge: Dictionary containing bridge parameters (bridge_id)
        output_path: Path where the scene XML file will be saved
    """
    bridge_id = bridge['bridge_id']
    
    # Find all OBJ files in the bridge folder
    base_dir = Path(__file__).parent.parent
    bridge_folder = base_dir / "Dataset" / "BridgeModels" / bridge_id
    
    # Get all OBJ files in the bridge folder
    obj_files = []
    if bridge_folder.exists():
        obj_files = sorted([f.name for f in bridge_folder.glob("*.obj")])
    
    # Start building XML content
    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_parts.append('<document>')
    xml_parts.append(f'    <scene id="TLS_{bridge_id}" name="TLS_{bridge_id}">')
    xml_parts.append('')
    
    # Add each component as a separate part (use absolute path so helios finds files on Linux)
    for idx, obj_file in enumerate(obj_files):
        obj_path = (bridge_folder / obj_file).resolve()
        xml_parts.append(f'        <part id="{idx}">')
        xml_parts.append('            <filter type="objloader">')
        xml_parts.append(f'                <param type="string" key="filepath" value="/Dataset/BridgeModels/{bridge_id}/{obj_file}" />')
        xml_parts.append('                <param type="string" key="up" value="z" />')
        xml_parts.append('            </filter>')
        xml_parts.append('        </part>')
    
    xml_parts.append('		')
    xml_parts.append('    </scene>')
    xml_parts.append('</document>')
    
    xml_content = '\n'.join(xml_parts)
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    print(f"Created scene file: {output_path} ({len(obj_files)} components)")

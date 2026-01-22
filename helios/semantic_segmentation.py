import os
from collections import defaultdict
import sys


def semantic_segmentation(input_dir, bridge_id):
    """
    Split a .xyz file into separate files based on the 9th column value.
    
    Args:
        input_file: Path to the input .xyz file
        output_dir: Directory to save output files (default: same as input file)
    """
    # Component mapping
    component_names = {
        0: "approach_slab",
        1: "back_wall",
        2: "deck",
        3: "piers",
        4: "railings",
        5: "wing_walls"
    }
    

    
    # Dictionary to store lines for each component
    component_files = defaultdict(list)
    
    # Read the input file and group lines by component
    print(f"Reading {input_dir}...")
    # Find all .xyz files in the input_dir
    xyz_files = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".xyz")
    ]
    if not xyz_files:
        print(f"No .xyz files found in {input_dir}")
        return

    all_lines = []  # To hold all scan points

    for input_file in xyz_files:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Collect line for the merged output
                all_lines.append(line)

                # Split the line by whitespace
                parts = line.split()
                
                # Get the 9th column (index 8)
                component_id = int(parts[8])
                component_files[component_id].append(line)
    
    # Write merged .xyz file containing all scans
    merged_output_file = os.path.join(input_dir, f"{bridge_id}_complete.xyz")
    with open(merged_output_file, 'w') as f_merged:
        f_merged.write('\n'.join(all_lines) + '\n')
    print(f"\nMerged all scans into {os.path.basename(merged_output_file)} ({len(all_lines)} points)")

    # Write separate files for each component
    print(f"\nWriting component files...")
    for component_id, lines in sorted(component_files.items()):
        output_file = os.path.join(input_dir, f"{component_names[component_id]}.xyz")
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        
        print(f"  - {os.path.basename(output_file)}: {len(lines)} points")
    
    print(f"\nDone! Split {sum(len(lines) for lines in component_files.values())} points into {len(component_files)} files.")


if __name__ == "__main__":
    

    input_dir = "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/"
    semantic_segmentation(input_dir)

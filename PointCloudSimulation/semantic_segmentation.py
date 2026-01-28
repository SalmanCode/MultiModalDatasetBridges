import os
import sys


def semantic_segmentation(component_files, segmented_bridge_dir):   
 

    component_names = {
        0: "approach_slab",
        1: "back_wall",
        2: "deck",
        3: "piers",
        4: "railings",
        5: "wing_walls"
    }



    # Writing separate files for each component
    print(f"  Writing segmented components...")
    for component_id, lines in sorted(component_files.items()):
        output_file = segmented_bridge_dir / f"{component_names[component_id]}.xyz"
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        
        print(f"    - {component_names[component_id]}.xyz: {len(lines):,} points")
    print(f"Split {sum(len(lines) for lines in component_files.values()):,} points into {len(component_files)} components")



if __name__ == "__main__":
    

    input_dir = "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/"
    semantic_segmentation(input_dir)

import os
import sys


def semantic_segmentation(component_files, segmented_bridge_dir, component_names=None):
    """Split a merged scan into per-component .xyz files."""
    if component_names is None:
        raise ValueError(
            "component_names is required: pass the sorted OBJ stems used to "
            "build the scene XML, e.g. "
            "{i: f.stem for i, f in enumerate(sorted(bridge_folder.glob('*.obj')))}"
        )

    # Writing separate files for each component
    print(f"  Writing segmented components...")
    for component_id, lines in sorted(component_files.items()):
        name = component_names.get(component_id, f"unknown_{component_id}")
        output_file = segmented_bridge_dir / f"{name}.xyz"
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print(f"    - {name}.xyz: {len(lines):,} points")
    print(f"Split {sum(len(lines) for lines in component_files.values()):,} points into {len(component_files)} components")


def component_names_for(bridge_folder):
    """The id -> name contract, derived from the scene XML's sorted OBJ order."""
    from pathlib import Path
    objs = sorted(Path(bridge_folder).glob("*.obj"))
    return {i: f.stem for i, f in enumerate(objs)}

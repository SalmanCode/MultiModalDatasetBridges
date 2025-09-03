"""
Colored Bridge Viewer
====================

A CadQuery script that generates bridge parts with different colors and displays them
using CadQuery's visualization tools. This script is designed to be run in cq-editor
or similar CadQuery visualization environments.

Features:
- Separate colored parts for each bridge component
- Interactive 3D visualization
- Color-coded parts for easy identification
- Export capabilities for individual parts

Color Scheme:
- Deck: Blue
- Railings: Red
- Wing Walls: Green
- Wing Wall Foundations: Yellow
- Retaining Walls: Purple
"""

import cadquery as cq
from cadquery import *
from dataclasses import dataclass

@dataclass
class BridgeViewerParams:
    """
    Bridge parameters for the colored viewer.
    """
    # Deck parameters
    deck_length: float = 50.0
    deck_width: float = 20.0
    deck_thickness: float = 1.0
    
    # Railing parameters
    railing_height: float = 4.0
    railing_thickness: float = 1
    slot_width: float = 2.0
    slot_height: float = 2.0
    slot_spacing: float = 10.0
    
    # Wing wall parameters
    wing_wall_cap_height: float = 1.0
    wing_wall_top_length: float = 4.0
    wing_wall_bottom_length: float = 2.0
    wing_wall_thickness: float = 1.0
    wing_wall_slot_height: float = 1.0
    wing_wall_side_length: float = 4.0
    wing_wall_slab_slot_length: float = 1.0

    # Foundation parameters
    foundation_paddings: float = 1.0
    foundation_foot_length: float = 10.0
    foundation_thickness: float = 1.0
    
    # Retaining wall parameters
    retaining_wall_thickness: float = 1.0

def make_deck_part(p: BridgeViewerParams) -> cq.Workplane:
    """Create the main deck slab."""
    deck = cq.Workplane("XY").box(
        p.deck_length, p.deck_width, p.deck_thickness,
        centered=(True, True, False)
    )
    return deck

def make_railings_part(p: BridgeViewerParams) -> cq.Workplane:
    """Create railings with rectangular slots."""
    num_of_slots = int(((p.deck_length/2)-2*p.slot_spacing)/(p.slot_spacing + p.slot_width))
    actual_railing_length = 2*(num_of_slots*(p.slot_spacing + p.slot_width)) + p.slot_spacing
    
    railing_block_left = cq.Workplane("XZ").box(
        actual_railing_length, p.railing_height, p.railing_thickness, 
        centered=(True, False, False)
    ).translate((0, p.deck_width/2, 0))
    
    # Create slots
    slot_blocks = cq.Workplane("XZ")
    for i in range(num_of_slots):
        slot_x = p.slot_spacing + i*(p.slot_spacing + p.slot_width)
        slot_block = cq.Workplane("XZ").box(
            p.slot_width, p.slot_height, 2, 
            centered=(False, True, False)
        ).translate((-p.slot_spacing/2 + slot_x, p.deck_width/2 + p.railing_thickness/2, p.deck_thickness + 2))
        slot_blocks = slot_blocks.union(slot_block)
    
    slot_blocks_other_half = slot_blocks.mirror("YZ")
    slot_blocks_all = slot_blocks.union(slot_blocks_other_half)
    
    railing_block_left_slot = railing_block_left.cut(slot_blocks_all)
    railing_block_right_slot = railing_block_left_slot.mirror("XZ")
    railings = railing_block_left_slot.union(railing_block_right_slot)
    
    return railings

def make_wing_walls_part(p: BridgeViewerParams) -> cq.Workplane:
    """Create wing walls."""
    x_origin = p.deck_length/2 - p.wing_wall_slab_slot_length
    z_origin = 0 - p.wing_wall_side_length
    
    points = [
        (x_origin, z_origin),
        (x_origin + p.wing_wall_bottom_length, z_origin),
        (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length - p.wing_wall_cap_height),
        (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length),
        (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length),
        (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_side_length),
        (x_origin, z_origin + p.wing_wall_side_length)
    ]
    
    left_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0))
    right_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0))
    left_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0)).mirror("YZ")
    right_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0)).mirror("YZ")

    wing_walls = left_front_wing_wall.union(right_front_wing_wall).union(left_back_wing_wall).union(right_back_wing_wall)
    return wing_walls

def make_wing_wall_foundations_part(p: BridgeViewerParams) -> cq.Workplane:
    """Create wing wall foundations."""
    foundation_width = p.wing_wall_bottom_length + 2*p.foundation_paddings
    p_1x = p.deck_length/2 - p.wing_wall_slab_slot_length - p.foundation_paddings
    
    p_1y = 0
    p_2x = p_1x + foundation_width
    p_2y = 0
    p_3x = p_2x
    p_3y = p.deck_width/2 - p.wing_wall_thickness - p.foundation_paddings
    p_4x = p_1x + p.foundation_foot_length
    p_4y = p_3y
    p_5x = p_4x
    p_5y = p.deck_width/2 + p.foundation_paddings
    p_6x = p_1x
    p_6y = p_5y
    
    points = [
        (p_1x, p_1y),
        (p_2x, p_2y),
        (p_3x, p_3y),
        (p_4x, p_4y),
        (p_5x, p_5y),
        (p_6x, p_6y)
    ]

    half_foundation = cq.Workplane("XY").polyline(points).close().extrude(p.foundation_thickness).translate((0, 0, -p.wing_wall_side_length))
    half_foundation_mirror = half_foundation.mirror("XZ")
    full_foundation = half_foundation.union(half_foundation_mirror)
    yz_mirror_foundation = full_foundation.mirror("YZ") 
    complete_foundation = full_foundation.union(yz_mirror_foundation)
    return complete_foundation

def make_retaining_wall_part(p: BridgeViewerParams) -> cq.Workplane:
    """Create retaining walls."""
    retaining_wall_front = cq.Workplane("YZ").box(
        p.deck_width, p.wing_wall_side_length, p.retaining_wall_thickness, 
        centered=(True, False, False)
    ).translate((p.deck_length/2 - p.wing_wall_slab_slot_length, 0, -p.wing_wall_side_length))
    retaining_wall_back = retaining_wall_front.mirror("YZ")
    return retaining_wall_front.union(retaining_wall_back)

def create_colored_bridge_viewer():
    """
    Create and display the colored bridge parts using CadQuery visualization.
    
    This function creates each bridge component with a distinct color and displays
    them using CadQuery's show_object function. Each part is shown separately
    with its own color for easy identification.
    """
    # Create bridge parameters
    p = BridgeViewerParams(
        deck_length=50.0,
        deck_width=20.0,
        deck_thickness=1.0,
        railing_height=5.0,
        railing_thickness=1.0,
        wing_wall_thickness=5.0,
        slot_width=3.0,
        slot_height=3.0,
        slot_spacing=2.0,
    )
    
    # Create individual parts
    deck = make_deck_part(p)
    railings = make_railings_part(p)
    wing_walls = make_wing_walls_part(p)
    wing_wall_foundations = make_wing_wall_foundations_part(p)
    retaining_walls = make_retaining_wall_part(p)
    
    # Display parts with colors (this will work in cq-editor)
    try:
        show_object  # type: ignore
        
        # Show deck in blue
        show_object(deck, name="deck", options={"color": (0, 0, 255, 0.8)})
        
        # Show railings in red
        show_object(railings, name="railings", options={"color": (255, 0, 0, 0.8)})
        
        # Show wing walls in green
        show_object(wing_walls, name="wing_walls", options={"color": (0, 255, 0, 0.8)})
        
        # Show foundations in yellow
        show_object(wing_wall_foundations, name="foundations", options={"color": (255, 255, 0, 0.8)})
        
        # Show retaining walls in purple
        show_object(retaining_walls, name="retaining_walls", options={"color": (128, 0, 128, 0.8)})
        
        print("Colored bridge parts displayed successfully!")
        print("Color scheme:")
        print("- Deck: Blue")
        print("- Railings: Red")
        print("- Wing Walls: Green")
        print("- Wing Wall Foundations: Yellow")
        print("- Retaining Walls: Purple")
        
        # Also show the complete assembly
        complete_bridge = deck.union(railings).union(wing_walls).union(wing_wall_foundations).union(retaining_walls)
        show_object(complete_bridge, name="complete_bridge", options={"color": (200, 200, 200, 0.6)})
        
        return {
            "deck": deck,
            "railings": railings,
            "wing_walls": wing_walls,
            "foundations": wing_wall_foundations,
            "retaining_walls": retaining_walls,
            "complete_bridge": complete_bridge
        }
        
    except NameError:
        print("This script is designed to run in cq-editor or similar CadQuery visualization environment.")
        print("The show_object function is not available in this environment.")
        return None

# Create the colored bridge viewer
bridge_parts = create_colored_bridge_viewer()

# Export individual parts if needed
if bridge_parts:
    import os
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    
    # Export individual colored parts
    for part_name, part_obj in bridge_parts.items():
        if part_name != "complete_bridge":  # Skip the complete assembly
            stl_file = os.path.join(out_dir, f"{part_name}.stl")
            cq.exporters.export(part_obj, stl_file)
            print(f"Exported {part_name} to {stl_file}")
    
    # Export complete assembly
    complete_stl = os.path.join(out_dir, "complete_bridge.stl")
    cq.exporters.export(bridge_parts["complete_bridge"], complete_stl)
    print(f"Exported complete bridge to {complete_stl}")

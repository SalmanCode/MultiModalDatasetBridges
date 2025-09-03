"""
Colored Bridge Assembly Generator
================================

A bridge model with separate colored parts:
- Deck (Blue)
- Railings (Red) 
- Wing Walls (Green)
- Wing Wall Foundations (Yellow)
- Retaining Walls (Purple)

This script demonstrates CadQuery assembly concepts:
- Separate part generation
- Color assignment for visualization
- Assembly combination
- Export to STL/OBJ with color information

CadQuery Assembly Basics:
- Each part is created separately
- Colors are assigned using show_object with color parameter
- Parts are combined using union operations
- Assembly maintains individual part properties
"""

import cadquery as cq
from cadquery import *
from dataclasses import dataclass

@dataclass
class BridgeAssemblyParams:
    """
    Bridge parameters dataclass for assembly generation.
    
    All dimensions are in arbitrary units (can be meters, feet, etc.).
    The coordinate system follows CadQuery convention:
    - X-axis: Bridge length (left to right)
    - Y-axis: Bridge width (front to back) 
    - Z-axis: Height (bottom to top)
    """
    # Deck parameters
    deck_length: float = 50.0      # Overall bridge length along X-axis
    deck_width: float = 20.0        # Overall bridge width along Y-axis
    deck_thickness: float = 1.0     # Deck thickness along Z-axis
    
    # Bridge span (distance between abutments)
    span_length: float = 20.0       # The actual bridge span length
    
    # Railing parameters
    railing_height: float = 4.0   # Height of railing above deck surface (Z-axis)
    railing_thickness: float = 1  # Thickness of railing bars (Y-axis)
    slot_width: float = 2.0
    slot_height: float = 2.0
    slot_spacing: float = 10.0
    
    # Abutment parameters
    abutment_height: float = 8.0    # Total height of abutment from ground to deck (Z-axis)
    abutment_width: float = 25.0    # Width of abutment along Y-axis
    abutment_length: float = 15.0   # Length of abutment extending from deck (X-axis)
    
    # Retaining wall parameters
    retaining_wall_height: float = 6.0      # Height of retaining wall (Z-axis)
    retaining_wall_thickness: float = 1.0   # Thickness of retaining wall (X-axis)
    
    # Wing wall parameters
    wing_wall_cap_height: float = 1.0      # Height of trapezoidal body (Z-axis)
    wing_wall_top_length: float = 4.0      # Length at top of trapezoid (Y-axis)
    wing_wall_bottom_length: float = 2.0   # Length at bottom of trapezoid (Y-axis)
    wing_wall_thickness: float = 1.0       # Thickness of wing wall (X-axis)
    wing_wall_slot_height: float = 1.0     # Height of cap slot for deck (Z-axis)
    wing_wall_side_length: float = 4.0      # Length of the side of the wing wall (Z-axis)
    wing_wall_slab_slot_length: float = 1.0 # Length of the slab slot in the wing wall (X-axis)

    # Foundation parameters
    foundation_paddings: float = 1.0
    foundation_foot_length: float = 10.0
    foundation_thickness: float = 1.0

def make_deck_part(p: BridgeAssemblyParams) -> cq.Workplane:
    """
    Create the main deck slab (Blue color).
    
    Returns:
        cq.Workplane: The deck as a 3D solid object
    """
    deck = cq.Workplane("XY").box(
        p.deck_length,      # Length along X-axis
        p.deck_width,       # Width along Y-axis
        p.deck_thickness,   # Height along Z-axis
        centered=(True, True, False)  # Center X,Y, bottom-align Z
    )
    return deck

def make_railings_part(p: BridgeAssemblyParams) -> cq.Workplane:
    """
    Create railings with rectangular slots (Red color).
    
    Returns:
        cq.Workplane: The railings as a 3D solid object
    """
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

def make_wing_walls_part(p: BridgeAssemblyParams) -> cq.Workplane:
    """
    Create wing walls (Green color).
    
    Returns:
        cq.Workplane: The wing walls as a 3D solid object
    """
    x_origin = p.deck_length/2 - p.wing_wall_slab_slot_length
    z_origin = 0 - p.wing_wall_side_length
    
    points = [
        (x_origin, z_origin), # bottom left point 1
        (x_origin + p.wing_wall_bottom_length, z_origin), # bottom right point 2
        (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length - p.wing_wall_cap_height), # right second point 3
        (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length), # right top point 4
        (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length), # top left point 5
        (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_side_length), # top left second point 6
        (x_origin, z_origin + p.wing_wall_side_length) # top left third point 7
    ]
    
    left_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0))
    right_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0))
    left_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0)).mirror("YZ")
    right_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0)).mirror("YZ")

    wing_walls = left_front_wing_wall.union(right_front_wing_wall).union(left_back_wing_wall).union(right_back_wing_wall)
    return wing_walls

def make_wing_wall_foundations_part(p: BridgeAssemblyParams) -> cq.Workplane:
    """
    Create wing wall foundations (Yellow color).
    
    Returns:
        cq.Workplane: The wing wall foundations as a 3D solid object
    """
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

def make_retaining_wall_part(p: BridgeAssemblyParams) -> cq.Workplane:
    """
    Create retaining walls (Purple color).
    
    Returns:
        cq.Workplane: The retaining walls as a 3D solid object
    """
    retaining_wall_front = cq.Workplane("YZ").box(
        p.deck_width, p.wing_wall_side_length, p.retaining_wall_thickness, 
        centered=(True, False, False)
    ).translate((p.deck_length/2 - p.wing_wall_slab_slot_length, 0, -p.wing_wall_side_length))
    retaining_wall_back = retaining_wall_front.mirror("YZ")
    return retaining_wall_front.union(retaining_wall_back)

def build_colored_bridge_assembly(p: BridgeAssemblyParams):
    """
    Build the complete bridge assembly with colored parts.
    
    This function:
    1. Creates each part separately
    2. Assigns different colors to each part
    3. Shows individual parts with colors
    4. Combines all parts into final assembly
    5. Exports to STL/OBJ format
    
    Color Scheme:
    - Deck: Blue
    - Railings: Red
    - Wing Walls: Green
    - Wing Wall Foundations: Yellow
    - Retaining Walls: Purple
    """
    # Create individual parts
    deck = make_deck_part(p)
    railings = make_railings_part(p)
    wing_walls = make_wing_walls_part(p)
    wing_wall_foundations = make_wing_wall_foundations_part(p)
    retaining_walls = make_retaining_wall_part(p)
    
    # Show individual parts with colors (for cq-editor visualization)
    try:
        show_object  # type: ignore
        show_object(deck, name="deck", options={"color": (0, 0, 255, 0.8)})  # Blue
        show_object(railings, name="railings", options={"color": (255, 0, 0, 0.8)})  # Red
        show_object(wing_walls, name="wing_walls", options={"color": (0, 255, 0, 0.8)})  # Green
        show_object(wing_wall_foundations, name="foundations", options={"color": (255, 255, 0, 0.8)})  # Yellow
        show_object(retaining_walls, name="retaining_walls", options={"color": (128, 0, 128, 0.8)})  # Purple
    except NameError:
        # Not running in cq-editor, skip individual part visualization
        pass
    
    # Combine all parts into final assembly
    bridge_assembly = deck.union(railings).union(wing_walls).union(wing_wall_foundations).union(retaining_walls)
    
    return bridge_assembly

if __name__ == "__main__":
    """
    Main execution block for the colored bridge assembly generator.
    
    This section:
    1. Creates bridge parameters with realistic dimensions
    2. Builds the complete bridge assembly with colored parts
    3. Exports to STL and OBJ formats
    4. Provides cq-editor integration with color visualization
    """
    # Create bridge parameters with realistic dimensions
    p = BridgeAssemblyParams(
        deck_length=50.0,        
        deck_width=20.0,           
        deck_thickness=1.0,        
        span_length=20.0,          
        railing_height=5.0,        
        railing_thickness=1.0,   
        abutment_height=8.0,       
        abutment_width=25.0,       
        abutment_length=15.0,      
        retaining_wall_height=6.0, 
        retaining_wall_thickness=1.0,
        wing_wall_cap_height=1.0,  
        wing_wall_top_length=4.0,  
        wing_wall_bottom_length=2.0,
        wing_wall_thickness=5.0,   
        wing_wall_slot_height=1.0,  
        wing_wall_side_length=4.0,  
        wing_wall_slab_slot_length=1.0,
        foundation_paddings=1.0,
        foundation_foot_length=10.0,
        foundation_thickness=1.0,
        slot_width=3.0,
        slot_height=3.0,
        slot_spacing=2.0,
    )
    
    # Build the complete bridge assembly with colored parts
    bridge_assembly = build_colored_bridge_assembly(p)
    
    # Export to multiple formats
    import os
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    
    # Export to STL and STEP formats
    stl_file = os.path.join(out_dir, "colored_bridge_assembly.stl")
    step_file = os.path.join(out_dir, "colored_bridge_assembly.step")
    
    cq.exporters.export(bridge_assembly, stl_file)
    cq.exporters.export(bridge_assembly, step_file)
    
    print("Colored bridge assembly exported successfully!")
    print(f"Files saved to: {out_dir}/")
    print("Color scheme:")
    print("- Deck: Blue")
    print("- Railings: Red") 
    print("- Wing Walls: Green")
    print("- Wing Wall Foundations: Yellow")
    print("- Retaining Walls: Purple")
    
    # Show final assembly in cq-editor (if available)
    try:
        show_object  # type: ignore
        show_object(bridge_assembly, name="colored_bridge_assembly")
    except NameError:
        # Not running in cq-editor, skip visualization
        pass

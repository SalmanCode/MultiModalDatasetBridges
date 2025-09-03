"""
Simplified CadQuery Bridge Generator
===================================

A simplified bridge model with:
- Main deck
- Rectangular abutments with stepped profiles at corners
- Railings on sides
- Wing walls with trapezoidal shape and caps

This script demonstrates CadQuery (CQ) concepts:
- Workplane operations (XY, XZ, YZ planes)
- Geometric primitives (box, polyline, extrude)
- Transformations (translate, offset, mirror)
- Boolean operations (union)
- Coordinate systems and positioning

CadQuery Basics:
- Workplane: A 2D plane in 3D space where operations happen
- Points: 3D coordinates (x, y, z) where x=left/right, y=forward/back, z=up/down
- Transformations: Move, rotate, or scale objects in 3D space
- Boolean operations: Combine solids using union, intersection, difference
"""

import cadquery as cq
from cadquery import *
from dataclasses import dataclass

@dataclass
class SimpleBridgeParams:
    """
    Bridge parameters dataclass for easy customization.
    
    All dimensions are in arbitrary units (can be meters, feet, etc.).
    The coordinate system follows CadQuery convention:
    - X-axis: Bridge length (left to right)
    - Y-axis: Bridge width (front to back) 
    - Z-axis: Height (bottom to top)
    """
    # Deck parameters
    deck_length: float = 50.0      # Overall bridge length along X-axis
    deck_width: float = 20.0        # Overall bridge width along Y-axis
    deck_thickness: float = 1.0     # Deck thickness along Z-axis (also wing wall slot height)
    
    # Bridge span (distance between abutments)
    span_length: float = 20.0       # The actual bridge span length
    
    # Railing parameters
    railing_height: float = 3.0     # Height of railing above deck surface (Z-axis)
    railing_thickness: float = 0.5  # Thickness of railing bars (Y-axis)
    
    # Abutment parameters (rectangular with stepped profile)
    abutment_height: float = 8.0    # Total height of abutment from ground to deck (Z-axis)
    abutment_width: float = 25.0    # Width of abutment along Y-axis
    abutment_length: float = 15.0   # Length of abutment extending from deck (X-axis)
    
    # Retaining wall parameters (behind abutments)
    retaining_wall_height: float = 6.0      # Height of retaining wall (Z-axis)
    retaining_wall_thickness: float = 1.0   # Thickness of retaining wall (X-axis)
    
    # Wing wall parameters (trapezoidal shape with cap)
    wing_wall_cap_height: float = 1.0      # Height of trapezoidal body (Z-axis)
    wing_wall_top_length: float = 4.0      # Length at top of trapezoid (Y-axis)
    wing_wall_bottom_length: float = 2.0   # Length at bottom of trapezoid (Y-axis)
    wing_wall_thickness: float = 1.0       # Thickness of wing wall (X-axis)
    wing_wall_slot_height: float = 1.0     # Height of cap slot for deck (Z-axis)
    wing_wall_side_length: float = 4.0      # Length of the side of the wing wall (Z-axis)
    wing_wall_slab_slot_length: float = 1.0 # Length of the slab slot in the wing wall (X-axis)


    # foundation parameters
    foundation_paddings: float = 1.0
    foundation_foot_length: float = 10.0
    foundation_thickness: float = 1.0

def make_deck(p: SimpleBridgeParams) -> cq.Workplane:
    """
    Create the main deck slab using CadQuery.
    
    CadQuery Explanation:
    - cq.Workplane("XY"): Creates a 2D workplane in the XY plane (horizontal)
    - .box(): Creates a rectangular prism (cuboid)
    - Parameters: (length, width, height, centered)
    - centered=(True, True, False): 
      * True: Center along X-axis (bridge length)
      * True: Center along Y-axis (bridge width) 
      * False: Align bottom to Z=0 (ground level)
    
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



def make_wing_walls(p: SimpleBridgeParams) -> cq.Workplane:
   
   ## this can be improved later. Right now there are multiple function calls and computation each time to calculate the wing wall. This can be done in one function call..
   # this makes the wing wall.
   # we will creat the front ones ( along x-axis ) . This will have a left and right wing wall (along y-axis).
   # then the back ones ( along x-axis ) will be the mirror of the front ones..
   x_origin = p.deck_length/2 - p.wing_wall_slab_slot_length
   z_origin = 0 - p.wing_wall_side_length
   
   points = [
           (x_origin, z_origin), #bottom left point 1
           (x_origin + p.wing_wall_bottom_length, z_origin), #bottom right point 2
           (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length - p.wing_wall_cap_height), #right second point 3,
           (x_origin + p.wing_wall_top_length + p.wing_wall_slab_slot_length, z_origin +p.wing_wall_slot_height + p.wing_wall_side_length), # right top point 4
           (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_slot_height + p.wing_wall_side_length), #top left point 5
           (x_origin + p.wing_wall_slab_slot_length, z_origin + p.wing_wall_side_length), #top left second point 6
           (x_origin, z_origin + p.wing_wall_side_length) #top left third point 7
       ]
   
   left_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0))
   right_front_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0))
   left_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(p.wing_wall_thickness).translate((0, p.deck_width/2, 0)).mirror("YZ")
   right_back_wing_wall = cq.Workplane("XZ").polyline(points).close().extrude(-p.wing_wall_thickness).translate((0, -p.deck_width/2, 0)).mirror("YZ")

   wing_walls = left_front_wing_wall.union(right_front_wing_wall).union(left_back_wing_wall).union(right_back_wing_wall)


   return wing_walls

def make_wing_wall_foundations(p: SimpleBridgeParams) -> cq.Workplane:

    foundation_width = p.wing_wall_bottom_length + 2*p.foundation_paddings
    ## this will create the foundation for the wing walls.
    p_1x = p.deck_length/2 - p.wing_wall_slab_slot_length - p.foundation_paddings
    p_1y = 0
    p_2x = p_1x
    p_2y = 0
    p_3x = p_1x + foundation_width
    p_3y = p.deck_width/2 - p.wing_wall_thickness - p.foundation_paddings
    p_4x = p_1x + p.foundation_foot_length
    p_4y = p_3y
    p_5x = p_4x
    p_5y = p.deck_width/2 + p.foundation_paddings
    p_6x = p_5x
    p_6y = p_1y

    
    
    
    

    points = [
        (p_1x, p_1y),
        (p_2x, p_2y),
        (p_3x, p_3y),
        (p_4x, p_4y),
        (p_5x, p_5y),
        (p_6x, p_6y)
    ]

    foundation = cq.Workplane("XY").polyline(points).close().extrude(p.foundation_thickness)
    return foundation

def build_simple_bridge(p: SimpleBridgeParams) -> cq.Workplane:
    """
    Build the complete simplified bridge by combining all components.
    
    CadQuery Explanation:
    - Each make_* function returns a cq.Workplane object
    - .union() combines multiple solids into one
    - Order of unions doesn't matter for final result
    - All components become part of single bridge solid
    
    Bridge Components:
    1. Deck: Main horizontal slab
    2. Railings: Safety barriers on deck edges
    3. Left Abutment: Stepped foundation on left side
    4. Right Abutment: Stepped foundation on right side
    5. Left Wing Walls: Trapezoidal walls extending from left abutment
    6. Right Wing Walls: Trapezoidal walls extending from right abutment
    
    Assembly Process:
    - Start with deck as base
    - Add railings on top
    - Add abutments at ends
    - Add wing walls extending from abutments
    - All components automatically align due to careful positioning
    
    Returns:
        cq.Workplane: Complete bridge as single 3D solid
    """
    # Create main components
    deck = make_deck(p)
    # create wing walls
    wing_wall_assembly = make_wing_walls(p)
    wing_wall_foundations = make_wing_wall_foundations(p)

    bridge = deck.union(wing_wall_assembly).union(wing_wall_foundations)
    return bridge


if __name__ == "__main__":
    """
    Main execution block for the bridge generator.
    
    This section:
    1. Creates bridge parameters with realistic dimensions
    2. Builds the complete bridge model
    3. Exports to STL and STEP formats
    4. Provides cq-editor integration
    5. Optional 3D visualization with coordinate axes
    
    CadQuery Export Explanation:
    - STL: 3D printable format (triangular mesh)
    - STEP: CAD exchange format (precise geometry)
    - Both formats preserve the exact 3D geometry
    """
    # Create bridge parameters with realistic dimensions
    p = SimpleBridgeParams(
        deck_length= 50.0,        
        deck_width=20.0,           # 20 units wide
        deck_thickness=1.0,        # 1 unit thick
        span_length=20.0,          # 20 unit span
        railing_height=3.0,        # 3 units high railings
        railing_thickness=0.5,     # 0.5 unit thick railings
        abutment_height=8.0,       # 8 units high abutments
        abutment_width=25.0,       # 25 units wide abutments
        abutment_length=15.0,      # 15 units long abutments
        retaining_wall_height=6.0, # 6 units high retaining walls
        retaining_wall_thickness=1.0, # 1 unit thick retaining walls
        wing_wall_cap_height=1.0,  # 1 unit high trapezoid
        wing_wall_top_length=4.0,  # 1 unit wide at top
        wing_wall_bottom_length=2.0, # 2 units wide at bottom
        wing_wall_thickness=5.0,   # 1 unit thick wing walls
        wing_wall_slot_height=1.0,  # 1 unit high slot for deck
        wing_wall_side_length=4.0,  # Length of the side of the wing wall (Z-axis)
        wing_wall_slab_slot_length=1.0, # Length of the slab slot in the wing wall (X-axis)
        foundation_paddings=1.0,
        foundation_foot_length=10.0,
        foundation_thickness=1.0
    )
    
    # Build the complete bridge model
    bridge = build_simple_bridge(p)
    
    # Export to multiple formats
    import os
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    
    # Export to STL (3D printing) and STEP (CAD) formats
    stl_file = os.path.join(out_dir, "simple_bridge.stl")
    step_file = os.path.join(out_dir, "simple_bridge.step")
    cq.exporters.export(bridge, stl_file)
    cq.exporters.export(bridge, step_file)
    
    print("Bridge exported successfully!")
    print(f"Files saved to: {out_dir}/")
    
    # cq-editor integration (optional)
    try:
        show_object  # type: ignore
        show_object(bridge, name="simple_bridge")
    except NameError:
        # Not running in cq-editor, skip visualization
        pass
    
    # 3D Visualization with coordinate axes (optional)
    ENABLE_3D_VISUALIZATION = True  # Set to False to disable
    
    if ENABLE_3D_VISUALIZATION:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from stl import mesh as stl_mesh
            
            def visualize_stl_with_coordinates(stl_file_path):
                """
                Visualize STL file with 3D coordinate axes.
                
                This function:
                1. Loads the STL file using numpy-stl
                2. Creates a 3D matplotlib plot
                3. Displays the mesh with coordinate axes
                4. Shows X, Y, Z directions with color coding
                5. Provides interactive 3D rotation
                
                Args:
                    stl_file_path (str): Path to the STL file to visualize
                """
                print(f"Loading STL file: {stl_file_path}")
                
                # Load the STL file
                mesh = stl_mesh.Mesh.from_file(stl_file_path)
                
                # Create figure and 3D axes
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get mesh bounds for coordinate axes scaling
                x_min, x_max = mesh.x.min(), mesh.x.max()
                y_min, y_max = mesh.y.min(), mesh.y.max()
                z_min, z_max = mesh.z.min(), mesh.z.max()
                
                # Calculate axis length (20% of model size)
                axis_length = max(x_max - x_min, y_max - y_min, z_max - z_min) 
                
                # Create coordinate axes
                origin = np.array([x_min, y_min, z_min])
                
                # X-axis (Red) - Bridge Length
                x_axis = np.array([[origin[0], origin[1], origin[2]], 
                                  [origin[0] + axis_length, origin[1], origin[2]]])
                ax.plot(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], 'r-', linewidth=3, label='X-axis (Length)')
                
                # Y-axis (Green) - Bridge Width  
                y_axis = np.array([[origin[0], origin[1], origin[2]], 
                                  [origin[0], origin[1] + axis_length, origin[2]]])
                ax.plot(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], 'g-', linewidth=3, label='Y-axis (Width)')
                
                # Z-axis (Blue) - Bridge Height
                z_axis = np.array([[origin[0], origin[1], origin[2]], 
                                  [origin[0], origin[1], origin[2] + axis_length]])
                ax.plot(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], 'b-', linewidth=3, label='Z-axis (Height)')
                
                # Add axis labels
                ax.text(origin[0] + axis_length*1.1, origin[1], origin[2], 'X (Length)', color='red', fontsize=12, fontweight='bold')
                ax.text(origin[0], origin[1] + axis_length*1.1, origin[2], 'Y (Width)', color='green', fontsize=12, fontweight='bold')
                ax.text(origin[0], origin[1], origin[2] + axis_length*1.1, 'Z (Height)', color='blue', fontsize=12, fontweight='bold')
                
                # Plot the mesh
                ax.add_collection3d(Poly3DCollection(mesh.vectors, alpha=0.7, facecolor='lightblue', edgecolor='black'))
                
                # Set axis labels and title
                ax.set_xlabel('X - Bridge Length', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y - Bridge Width', fontsize=12, fontweight='bold')
                ax.set_zlabel('Z - Bridge Height', fontsize=12, fontweight='bold')
                ax.set_title('Bridge Model with 3D Coordinate System', fontsize=14, fontweight='bold')
                
                # Set equal aspect ratio
                ax.set_box_aspect([1, 1, 1])
                
                # Add legend
                ax.legend()
                
                # Add coordinate information
                info_text = f"""
                Model Dimensions:
                X (Length): {x_max - x_min:.1f} units
                Y (Width): {y_max - y_min:.1f} units  
                Z (Height): {z_max - z_min:.1f} units
                
                Coordinate System:
                • X-axis (Red): Bridge length direction
                • Y-axis (Green): Bridge width direction
                • Z-axis (Blue): Bridge height direction
                
                Interactive Controls:
                • Rotate: Click and drag
                • Zoom: Scroll wheel
                • Pan: Right-click and drag
                """
                
                plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
                
                print("3D visualization ready! Use mouse to rotate, zoom, and pan.")
                plt.show()
                
            # Visualize the generated STL file
            visualize_stl_with_coordinates(stl_file)
            
        except ImportError as e:
            print(f"3D visualization requires additional packages: {e}")
            print("To enable 3D visualization, install: pip install matplotlib numpy-stl")
        except Exception as e:
            print(f"3D visualization error: {e}")
            print("Continuing without visualization...")

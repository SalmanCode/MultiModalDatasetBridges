import cadquery as cq
from .model_config import BridgeConfig
from dataclasses import asdict
from typing import Optional, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)






class BridgeModel:
    def __init__(self, config: BridgeConfig):
        self.config = config
        

    def make_deck(self) -> cq.Workplane:

        if self.config.bridge_type == "beam_slab":
            return self.make_beam_slab_deck()
        elif self.config.bridge_type == "box_girder":
            return self.make_box_girder_deck()
        else:
            raise ValueError(f"Invalid bridge type: {self.config.bridge_type}")

    def compute_tee_girder_spacing(self, width_m: float, min_spacing_m: float = 3.5) -> tuple[int, float]:
        
        num_of_girders = max(2, round(width_m / min_spacing_m))
        
        spacing = width_m / num_of_girders
        return num_of_girders, round(spacing, 1)

    def compute_box_girder_spacing(self) -> tuple[int, float]:
        
        ratio = self.config.depth_of_girder / self.config.width_m
        if ratio >= 1/6 and ratio <= 1/5:
            num_of_cells = 1
        elif ratio < 1/6:
            num_of_cells = 2
        box_width = (self.config.width_m - 4) / num_of_cells # here 4 is assumed that the l1 that is the length left out on each side of the box should be between 2 to 4 meters
        return num_of_cells, box_width

    def make_box_girder_deck(self) -> cq.Workplane:
        """ This makes deck with box girders"""
        
        deck_length = self.config.total_length_m
        total_width = self.config.width_m
        num_spans = self.config.num_spans
        span_length = self.config.span_m
        depth_of_girder = self.config.depth_of_girder



        # Basic geometric assumptions (meters)
        top_slab_thk = getattr(self.config, "top_slab_thk", 0.20)
        bottom_slab_thk = getattr(self.config, "bottom_slab_thk", 0.15)
        web_thk = getattr(self.config, "web_thk", 0.5)

        ratio = total_width / depth_of_girder
        # logger.info(f"Ratio: {ratio}")

        inner_edge_haunch = 0.4     
        outer_edge_haunch = 0.2     


        #logger.info(f"Box depth: {depth_of_girder}")
        
        
        num_cells, box_width = self.compute_box_girder_spacing()
        inner_cell_width = box_width - 2 * web_thk # because we are subtracting from the total box width the web thickness on both sides
        outer_cell_height = self.config.depth_of_girder
        inner_cell_height = outer_cell_height - top_slab_thk - bottom_slab_thk

        top_slab = (
            cq.Workplane("YZ")
            .box(total_width, top_slab_thk, deck_length, centered=(True, False, True)))


        boxes = cq.Workplane("YZ")
        #logger.info(f"total width: {total_width}")
        
        for i in range(num_cells):

            box_center_y = - ((num_cells - 1) * box_width / 2) + box_width * i  
            #logger.info(f"Box center y: {box_center_y}")
            

            # outer shell
            outer = cq.Workplane("YZ").box(box_width, outer_cell_height, deck_length, centered=(True, True, True)).translate((0, box_center_y, -depth_of_girder/2))
            inner = cq.Workplane("YZ").box(inner_cell_width, inner_cell_height, deck_length, centered=(True, True, True)).translate((0, box_center_y, -depth_of_girder/2))
            shell = outer.cut(inner)
            
            boxes = boxes.union(shell)

        # now we need to make the haunches on the outer edges of the boxes
        p1 = (box_width, -inner_edge_haunch)
        p2 = (self.config.width_m/2, -inner_edge_haunch + outer_edge_haunch)
        p3 = (p2[0], 0)
        p4 = (p1[0], 0)

        points = [p1, p2, p3, p4]

        haunch = cq.Workplane("YZ").polyline(points).close().extrude(deck_length).translate((-deck_length/2, 0, 0))
        haunch_mirror = haunch.mirror("ZX")
        haunch = haunch.union(haunch_mirror)
        boxes = boxes.union(haunch)

        bridge_core = top_slab.union(boxes)

        return bridge_core

    

    def make_beam_slab_deck(self) -> cq.Workplane:
        """ This makes beam slab or more like Tee girders """

        deck_length = self.config.total_length_m
        width = self.config.width_m
        deck_thickness = getattr(self.config, "deck_thickness", 0.3)
        depth_of_girder = self.config.depth_of_girder

        girder_thickness = 0.5
        minimum_girder_spacing = 3.5
        chamfer_radius = 0.1
        num_of_girders, girder_distance = self.compute_tee_girder_spacing(width, minimum_girder_spacing)

        # Ratio of width to girder height. This should be less more than 1/6 ideally.
        ratio = width / depth_of_girder
        #logger.info(f"Ratio: {ratio}")

        deck_slab = cq.Workplane("XY").box(deck_length, width, deck_thickness, centered=(True, True, False))
        
        girders = cq.Workplane("YZ")
        for i in range(num_of_girders):
            girder_y_position = -(girder_distance * (num_of_girders - 1) / 2)  + girder_distance * i
            girder = cq.Workplane("YZ").rect(girder_thickness, -depth_of_girder, centered=(True, False)).extrude(deck_length)
            
            girder = girder.translate((-deck_length / 2, girder_y_position , 0)) # remember tranlsation always happens in global coordinates so x, y, z.
            girders = girders.union(girder)

        
        bridge_core = (
            deck_slab.union(girders)
            .faces("-Z")          # bottom faces
            .edges("|X")          # edges parallel to Y (girder webs)
            .chamfer(chamfer_radius)
        )
        
        return bridge_core
        
    def make_approach_slabs(self) -> Optional[cq.Workplane]:

        approach_slab_thickness = self.config.deck_thickness
        approach_slab_width = self.config.width_m
        approach_slab_length = 10.0
        approach_slab = cq.Workplane("XY").box(approach_slab_length, approach_slab_width, approach_slab_thickness, centered=(False, True, False))
        approach_slab = approach_slab.translate((self.config.total_length_m/2, 0, 0))
        approach_slab_mirror = approach_slab.mirror("YZ")
        approach_slab = approach_slab.union(approach_slab_mirror)
        return approach_slab

    def make_railings(self) -> Optional[cq.Workplane]:
        """ this makes the railings for the bridge if safety is required"""
        """ this can return None if safety is not required"""

        railing_pole_height = 1.0 # this is taken from RZ standards
        railing_pole_distance = 2.5
        railing_pole_side_length = 0.07 # this is assuming that the pole is a square in xy plane with 7 cm side length
        deck_length = self.config.total_length_m 
        

        num_of_poles = max(2, int(deck_length / (railing_pole_distance)))
        

        railing_poles = cq.Workplane("XY")

        # we can take the distance between the 1st pole and the last pole to get the total length of the bar which would be less than the span length. 
        # easy way is to just minus the distance of half poles from both sides.


        for i in range(num_of_poles):
            pole_x_position = - ( railing_pole_distance * (num_of_poles - 1) /2) + railing_pole_distance * i
            pole = cq.Workplane("XY").box(railing_pole_side_length, railing_pole_side_length, railing_pole_height, centered=(True, True, False)).translate((pole_x_position, self.config.width_m / 2 - railing_pole_side_length / 2, self.config.deck_thickness))
            railing_poles = railing_poles.union(pole)

        
        railing_poles_mirror = railing_poles.mirror("XZ")
        railing_poles = railing_poles.union(railing_poles_mirror)


        # now we add railing bars between the poles

        num_of_bars = 3
        distance_between_bars = railing_pole_height / num_of_bars
        bars = cq.Workplane("XY")
        for i in range(num_of_bars):
            bar_z_position = distance_between_bars * (i + 1)
            bar = cq.Workplane("XY").box(deck_length - railing_pole_distance, railing_pole_side_length, railing_pole_side_length, centered=(True, True, True)).translate((0, self.config.width_m / 2 - railing_pole_side_length / 2, bar_z_position  + self.config.deck_thickness))
            bars = bars.union(bar)
        
        bars_mirror = bars.mirror("XZ")
        bars = bars.union(bars_mirror)

        return railing_poles.union(bars)

    def compute_pier_positions_along_length(self) -> list[float]:
        
        num_of_spans= self.config.num_spans
        interior_span_length = round(self.config.total_length_m / (num_of_spans - 0.6), 1)
        end_span_length = round(interior_span_length * 0.7, 1)

        #create a list of spans
        spans = [end_span_length] + [interior_span_length] * (num_of_spans-2) + [end_span_length]
        pier_positions = []
        pos = 0.0
        for span in spans[:-1]:
            pos += span

            pier_positions.append(round(pos, 1))

        normalised_pier_positions = [round(p - (self.config.total_length_m / 2), 1) for p in pier_positions]
        
        return normalised_pier_positions


    def make_piers(self) -> Optional[cq.Workplane]:


        bridge_clearance_height = getattr(self.config, "bridge_clearance_height", 5.0) # this is the minimum clearance height from the girder to the ground
         
    
        # num of piers in x direction is same for both types of piers
        num_of_piers_x = self.config.number_of_piers_along_length # this is the number of piers in the x direction
        pier_positions_x = self.compute_pier_positions_along_length() 

        if self.config.pier_type == "multicolumn":
            
            
            
            num_of_piers_y = self.config.number_of_piers_across_width
            pier_spacing_y = self.config.width_m / num_of_piers_y
            

            # Multi column piers require a pier cap
            piers = cq.Workplane("XY")
            pier_caps = cq.Workplane("XY")
            cap_height = 0.5     # Right now assuming that the cap height is 1 meter
            cap_thickness = self.config.radius_of_pier * 2


            # Generating piers columns
            for i in range(num_of_piers_x):
                pier_position_x = pier_positions_x[i]
                for j in range(num_of_piers_y):
                    pier_position_y = -((pier_spacing_y) * (num_of_piers_y - 1)/2) + pier_spacing_y * j
                    if self.config.pier_cross_section == "circular":
                        pier = cq.Workplane("XY").circle(self.config.radius_of_pier).extrude(-bridge_clearance_height).translate((pier_position_x, pier_position_y, - self.config.depth_of_girder - cap_height))
                    elif self.config.pier_cross_section == "rectangular":
                        pier = cq.Workplane("XY").rect(self.config.radius_of_pier, self.config.radius_of_pier).extrude(-bridge_clearance_height).translate((pier_position_x, pier_position_y, - self.config.depth_of_girder - cap_height))
                    piers = piers.union(pier)
                pier_cap = self.make_prismatic_pier_caps(cap_height, cap_thickness).translate((pier_position_x, 0, -self.config.depth_of_girder - cap_height))
                pier_caps = pier_caps.union(pier_cap)
            piers = piers.union(pier_caps)
        

        if self.config.pier_type == "hammer_head":


            piers = cq.Workplane("YZ")
            num_of_piers_y = self.config.number_of_piers_across_width
            _, box_width = self.compute_box_girder_spacing()
            pier_spacing_y = self.config.width_m / num_of_piers_y # this is to calculate the spacing between the piers accross the width of the bridge. This will be the same as the spacing between the cells in the box girder.

            #Creating polygon geometry for hammer head piers
            polygon_height = 1 # Literature based. Check notion repository for more details.
            polygon_slant_height = 1 # Literature based. Check notion repository for more details.
            if self.config.pier_cross_section == "circular":
                polygon_lower_width = 2*self.config.radius_of_pier
            elif self.config.pier_cross_section == "rectangular":
                polygon_lower_width = 1.8 # this is in case of rectangular piers for hammerhead. from literature. Check notion repository for more details.
            
            p1 = (box_width/2, 0)
            p2 = (p1[0], - polygon_height)
            p3 = (polygon_lower_width/2, -polygon_slant_height - polygon_height)
            p4 = (0, p3[1])
            p5 = (0 , 0)
            cap_height = polygon_height + polygon_slant_height # this will be the height that is the sum of two lines one straigt and one slanted.
            pier_thickness = 1.0 # assumed thickness of the pier column

            points = [p1, p2, p3, p4, p5]
            for i in range(num_of_piers_x):
                pier_position_x = pier_positions_x[i]
                for j in range(num_of_piers_y):
                    # we will calculate the piers accross the width of the bridge
                    pier_position_y = -((pier_spacing_y) * (num_of_piers_y - 1)/2) + pier_spacing_y * j

                    # first making rectungular column
                    if self.config.pier_cross_section == "circular":
                        pier_column = cq.Workplane("XY").circle(self.config.radius_of_pier).extrude(-bridge_clearance_height)
                    elif self.config.pier_cross_section == "rectangular":
                        pier_column = cq.Workplane("YZ").rect(polygon_lower_width, -bridge_clearance_height).extrude(pier_thickness, both=True)
                   
                    pier_column = pier_column.translate((pier_position_x, pier_position_y, - self.config.depth_of_girder - cap_height ))

                    # then making the hammer head shape ( pier cap)
                    pier_cap = cq.Workplane("YZ").polyline(points).close().extrude(pier_thickness, both=True)
                    pier_cap_mirror = pier_cap.mirror("XZ")
                    pier_cap = pier_cap.union(pier_cap_mirror)
                    pier_cap = pier_cap.translate((pier_position_x, pier_position_y, - self.config.depth_of_girder ))
                    piers = piers.union(pier_column)
                    piers = piers.union(pier_cap)

        return piers
        

    
    def make_prismatic_pier_caps(self, cap_height: float, cap_thickness: float) -> Optional[cq.Workplane]:
     
        """ This makes the prismatic pier caps """

        cap_width = self.config.width_m     # width along traffic
        pier_cap = cq.Workplane("YZ").box(cap_width, cap_height, cap_thickness, centered=(True, False, True))
        return pier_cap
        
    def make_wing_walls(self) -> Optional[cq.Workplane]:
        """Create wing walls if not missing."""

        # Geometric parameters (in meters)
        wing_wall_thickness = getattr(self.config, "wing_wall_thickness", 0.5)
        bridge_seating_height = self.config.depth_of_girder
        bridge_seating_width = 2
        wing_wall_cap_height = 0.5 * self.config.bridge_clearance_height
        wing_wall_top_length = 4.0
        wing_wall_bottom_length = 2.0
        wing_wall_side_length = self.config.bridge_clearance_height

        deck_length = self.config.total_length_m
        deck_width = self.config.width_m

        wing_wall_slab_slot_length = bridge_seating_width
        wing_wall_slot_height = bridge_seating_height

        x_origin = deck_length / 2 - bridge_seating_width
        z_origin = - self.config.bridge_clearance_height - self.config.depth_of_girder
        # Define points for the wing wall profile in the XZ-plane
        p1 = (x_origin, z_origin)
        p2 = (x_origin + wing_wall_bottom_length, z_origin)
        p3 = (
            x_origin + wing_wall_top_length + bridge_seating_width,
            z_origin + bridge_seating_height + wing_wall_side_length - wing_wall_cap_height,
        )
        p4 = (
            p3[0],
            z_origin + bridge_seating_height + wing_wall_side_length,
        )
        p5 = (
            x_origin + bridge_seating_width,
            p4[1],
        )
        p6 = (
            p5[0],
            p5[1] - bridge_seating_height,
        )
        p7 = (x_origin, z_origin + wing_wall_side_length)

        points = [p1, p2, p3, p4, p5, p6, p7]

        def create_wing_wall(translate_y: float, thickness: float, mirror_yz: bool = False) -> cq.Workplane:
            wall = (
                cq.Workplane("XZ")
                .polyline(points)
                .close()
                .extrude(thickness)
                .translate((0, translate_y, 0))
            )
            if mirror_yz:
                wall = wall.mirror("YZ")
            return wall

        left_front = create_wing_wall(deck_width / 2, wing_wall_thickness, mirror_yz=False)
        right_front = create_wing_wall(-deck_width / 2, -wing_wall_thickness, mirror_yz=False)
        left_back = create_wing_wall(deck_width / 2, wing_wall_thickness, mirror_yz=True)
        right_back = create_wing_wall(-deck_width / 2, -wing_wall_thickness, mirror_yz=True)

        wing_walls = (
            left_front
            .union(right_front)
            .union(left_back)
            .union(right_back)
        )

        return wing_walls

    def make_back_walls(self) -> Optional[cq.Workplane]:

        back_wall_thickness = 2
        back_wall = cq.Workplane("YZ").box(
            self.config.width_m - 2 * self.config.wing_wall_thickness,
            self.config.bridge_clearance_height,
            back_wall_thickness,
            centered=(True, False, False)
        ).translate((self.config.total_length_m/2 - back_wall_thickness, 0, -self.config.depth_of_girder - self.config.bridge_clearance_height))       
        # Back retaining wall
        wall_mirror = back_wall.mirror("YZ")  
    
        
        # this adds a thinner wall at the end of back wall to close the end of girders.
        thinner_wall = cq.Workplane("YZ").box(self.config.width_m - 2 * self.config.wing_wall_thickness, self.config.bridge_clearance_height + self.config.depth_of_girder, 0.5, centered=(True, False, False))
        thinner_wall = thinner_wall.translate((self.config.total_length_m/2, 0, -self.config.bridge_clearance_height - self.config.depth_of_girder))    
        thinner_wall_mirror = thinner_wall.mirror("YZ")
        thinner_wall_combined = thinner_wall.union(thinner_wall_mirror)
        
        return back_wall.union(wall_mirror).union(thinner_wall_combined)

    
    def build_bridge(self, with_components: bool = False) -> cq.Workplane | Dict[str, cq.Workplane]:
        """
        Build the bridge geometry.

        Args:
            as_components: When True, return each component individually instead of a
                single boolean union.

        Returns:
            Either a combined cq.Workplane or a dict of component solids.
        """
        logger.info(f"****BRIDGE {self.config.bridge_id} type: {self.config.bridge_type}****")
        logger.info(f"Building bridge {self.config.bridge_id} with config: {asdict(self.config)}")
        
        components: Dict[str, cq.Workplane | None] = {
            "deck": self.make_deck(),
            "approach_slabs": self.make_approach_slabs(),
            "railings": self.make_railings(),
            "piers": self.make_piers(),
            "wing_walls": self.make_wing_walls(),
            "back_walls": self.make_back_walls(),
        }

        filtered = {name: solid for name, solid in components.items() if solid is not None}

        bridge: cq.Workplane | None = None
        for solid in filtered.values():
            bridge = solid if bridge is None else bridge.union(solid)

        if bridge is None:
            raise ValueError("No bridge components were generated; check configuration.")

        if with_components:
            return filtered, bridge
        else:
            return bridge

        
       

import cadquery as cq
from config import BridgeConfig
from dataclasses import asdict
from typing import Optional
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

    def compute_box_girder_spacing(self,width_m: float, min_spacing_m: float = 5.0) -> int:
        num_of_cells = max(1, round(width_m / (min_spacing_m + 1)) - 1)
        return num_of_cells

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
        logger.info(f"Ratio: {ratio}")

        inner_cell_gap = 5.0  # desired clear width between webs
        edge_haunch = 0.4     # cantilever each side for parapets

        logger.info(f"Box depth: {depth_of_girder}")
        
        outer_width = inner_cell_gap + web_thk
        num_cells = self.compute_box_girder_spacing(total_width, inner_cell_gap)
  
        top_slab = (
            cq.Workplane("YZ")
            .box(total_width, top_slab_thk, deck_length, centered=(True, False, True)))


        boxes = cq.Workplane("YZ")
        logger.info(f"total width: {total_width}")
        
        for i in range(num_cells):

            box_center_y = - ((num_cells - 1) * outer_width / 2) + outer_width * i  
            logger.info(f"Box center y: {box_center_y}")
            

            # outer shell
            outer = cq.Workplane("YZ").rect(outer_width, depth_of_girder, centered=(True, True)).extrude(deck_length).translate((-deck_length / 2, box_center_y, -depth_of_girder/2))
            inner = cq.Workplane("YZ").rect(outer_width - 2 * web_thk, depth_of_girder - top_slab_thk - 0.15 - bottom_slab_thk, centered=(True, True)).extrude(deck_length).translate((-deck_length / 2, box_center_y, -depth_of_girder/2 - 0.15))
            shell = outer.cut(inner)
            
            boxes = boxes.union(shell)

        
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
        logger.info(f"Ratio: {ratio}")

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
        

    def make_railings(self) -> Optional[cq.Workplane]:
        """ this makes the railings for the bridge if safety is required"""
        """ this can return None if safety is not required"""

        railing_pole_height = 1.0
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

    def compute_pier_positions(self) -> list[float]:
        
        num_of_spans= self.config.num_spans
        interior_span_length = round(self.config.total_length_m / (num_of_spans - 0.6), 1)
        end_span_length = round(interior_span_length * 0.7, 1)

        #create a list of spans
        spans = [end_span_length] + [interior_span_length] * (num_of_spans-2) + [end_span_length]

        logger.info(f"spans: {spans}")
        print(f"spans: {spans}")
        pier_positions = []
        pos = 0.0
        for span in spans[:-1]:
            pos += span

            pier_positions.append(round(pos, 1))

        normalised_pier_positions = [round(p - (self.config.total_length_m / 2), 1) for p in pier_positions]
            
        
        print(f"pier positions: {normalised_pier_positions}")
        return normalised_pier_positions


    def make_piers(self) -> Optional[cq.Workplane]:

        bridge_clearance_height = getattr(self.config, "bridge_clearance_height", 5.0)
         # this is the minimum clearance height from the girder to the ground
        pier_positions = self.compute_pier_positions()

        # currently implementing circular piers
        num_of_piers = self.config.num_spans - 1
        
        piers = cq.Workplane("XY")
        pier_caps = cq.Workplane("XY")


        num_of_piers_per_lane = self.config.number_of_piers_per_lane
        num_of_lanes = self.config.lanes
        num_of_piers_longitude = num_of_piers_per_lane * num_of_lanes
        
        pier_spacing_y = self.config.width_m / num_of_piers_longitude
        logger.info(f"pier spacing y: {pier_spacing_y}")

        cap_height = 0.5     # Right now assuming that the cap height is 1 meter
        cap_thickness = self.config.radius_of_pier * 2

        if self.config.pier_type == "multicolumn":
            for i in range(num_of_piers):
                pier_position_x = pier_positions[i]
                for j in range(num_of_piers_longitude):
                    pier_position_y = -((pier_spacing_y) * (num_of_piers_longitude - 1)/2) + pier_spacing_y * j
                    pier = cq.Workplane("XY").circle(self.config.radius_of_pier).extrude(-bridge_clearance_height).translate((pier_position_x, pier_position_y, - self.config.depth_of_girder - cap_height))
                    piers = piers.union(pier)
                pier_cap = self.make_prismatic_pier_caps(cap_height, cap_thickness).translate((pier_position_x, 0, -self.config.depth_of_girder - cap_height))
                pier_caps = pier_caps.union(pier_cap)
            piers = piers.union(pier_caps)
        return piers
        

    
    def make_prismatic_pier_caps(self, cap_height: float, cap_thickness: float) -> Optional[cq.Workplane]:
     
        """ This makes the prismatic pier caps """

        cap_width = self.config.width_m     # width along traffic
        pier_cap = cq.Workplane("YZ").box(cap_width, cap_height, cap_thickness, centered=(True, False, True))
        return pier_cap
        
    def make_wing_walls(self) -> Optional[cq.Workplane]:
        """Create wing walls if not missing"""

        wing_wall_thickness = 0.5
        bridge_seating_height = self.config.depth_of_girder
        bridge_seating_width = 0.5
        wing_wall_cap_height = 0.5
        wing_wall_top_length = 2.0
        wing_wall_side_length = self.config.bridge_clearance_height
        wing_wall_bottom_length = 2.0

        deck_length = self.config.total_length_m
        deck_width = self.config.width_m

        wing_wall_slab_slot_length = bridge_seating_width
        wing_wall_slot_height = bridge_seating_height

        x_origin = deck_length / 2 - bridge_seating_width
        z_origin = -self.config.bridge_clearance_height

        points = [
            (x_origin, z_origin),
            (x_origin + wing_wall_bottom_length, z_origin),
            (x_origin + wing_wall_top_length + wing_wall_slab_slot_length,
             z_origin + wing_wall_slot_height + wing_wall_side_length - wing_wall_cap_height),
            (x_origin + wing_wall_top_length + wing_wall_slab_slot_length,
             z_origin + wing_wall_slot_height + wing_wall_side_length),
            (x_origin + wing_wall_slab_slot_length,
             z_origin + wing_wall_slot_height + wing_wall_side_length),
            (x_origin + wing_wall_slab_slot_length,
             z_origin + wing_wall_side_length),
            (x_origin, z_origin + wing_wall_side_length)
        ]

        left_front_wing_wall = (
            cq.Workplane("XZ")
            .polyline(points)
            .close()
            .extrude(wing_wall_thickness)
            .translate((0, deck_width / 2, 0))
        )

        right_front_wing_wall = (
            cq.Workplane("XZ")
            .polyline(points)
            .close()
            .extrude(-wing_wall_thickness)
            .translate((0, -deck_width / 2, 0))
        )

        left_back_wing_wall = (
            cq.Workplane("XZ")
            .polyline(points)
            .close()
            .extrude(wing_wall_thickness)
            .translate((0, deck_width / 2, 0))
            .mirror("YZ")
        )

        right_back_wing_wall = (
            cq.Workplane("XZ")
            .polyline(points)
            .close()
            .extrude(-wing_wall_thickness)
            .translate((0, -deck_width / 2, 0))
            .mirror("YZ")
        )

        wing_walls = (
            left_front_wing_wall
            .union(right_front_wing_wall)
            .union(left_back_wing_wall)
            .union(right_back_wing_wall)
        )

        return wing_walls

    def build_bridge(self) -> cq.Workplane:
        """Build the complete bridge with optional components"""
        logger.info(f"Building bridge {self.config.bridge_id} with config: {asdict(self.config)}")
        
        # Start with deck
        bridge = self.make_deck()
        bridge = bridge.union(self.make_railings())
        bridge = bridge.union(self.make_piers())
        bridge = bridge.union(self.make_wing_walls())
       

        
        return bridge
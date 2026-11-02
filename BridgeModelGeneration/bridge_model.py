import cadquery as cq
from .model_config import BridgeConfig
from dataclasses import asdict
from typing import Optional, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)






# scene-only components, dropped from the point cloud
NON_STRUCTURE_COMPONENTS = ("ground",)

# trapezoidal column (blade) proportions
TRAPEZOID_TOP_OVER_THICKNESS = 3.5   # real 2.9 - 4.7
TRAPEZOID_TOP_OVER_BASE = 1.65       # real 1.55 - 1.95, wider at the top

# spill-through abutment
SPILL_SEAT_HEIGHT = (0.8, 1.5)       # m, clamped to a share of the abutment height
SPILL_REVETMENT = 0.5                # m, vertical thickness of the slope slab


class BridgeModel:
    def __init__(self, config: BridgeConfig):
        self.config = config

    # z = 0 is the deck soffit
    @property
    def z_soffit(self) -> float:
        """Lowest structural surface: girder / box soffit."""
        return -self.config.depth_of_girder

    @property
    def z_ground(self) -> float:
        """Ground under the opening. Everything standing on the ground must
        start here -- piers AND abutments -- or they disagree by cap_height."""
        return self.z_soffit - self.config.bridge_clearance_height

    @property
    def road_z(self) -> float:
        """Top of the running surface: the top slab sits on the z = 0 soffit
        datum for both deck types."""
        return self.config.deck_thickness


    def make_deck(self) -> cq.Workplane:

        if self.config.bridge_type == "slab":
            return self.make_slab_deck()
        elif self.config.bridge_type == "girder":
            return self.make_girder_deck()
        else:
            raise ValueError(f"Invalid bridge type: {self.config.bridge_type}")

    def compute_tee_girder_spacing(self, width_m: float, min_spacing_m: float = 3.5) -> tuple[int, float]:
        
        num_of_girders = max(2, round(width_m / min_spacing_m))
        
        spacing = width_m / num_of_girders
        return num_of_girders, round(spacing, 1)

    def make_slab_deck(self) -> cq.Workplane:
        """Solid slab deck from the soffit to the road surface."""
        thickness = self.config.deck_thickness + self.config.depth_of_girder
        return cq.Workplane("XY").box(
            self.config.total_length_m, self.config.width_m, thickness,
            centered=(True, True, False),
        ).translate((0, 0, self.z_soffit))

    def make_girder_deck(self) -> cq.Workplane:
        """Top slab over the whole structure, T-beam webs under the opening."""

        deck_length = self.config.inner_deck_length_m
        total_length = self.config.total_length_m
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

        deck_slab = cq.Workplane("XY").box(total_length, width, deck_thickness, centered=(True, True, False)) # this will be total lenght because we want to conver the wingwall
        
        girders = cq.Workplane("YZ")
        for i in range(num_of_girders):
            girder_y_position = -(girder_distance * (num_of_girders - 1) / 2)  + girder_distance * i
            girder = cq.Workplane("YZ").rect(girder_thickness, -depth_of_girder, centered=(True, False)).extrude(deck_length)
            # Chamfer the solid girder first; fused deck/compound solids fail in OCC.
            girder = (
                girder
                .faces("-Z")          # bottom faces
                .edges("|X")          # long soffit edges along the girder
                .chamfer(chamfer_radius)
            )
            girder = girder.translate((-deck_length / 2, girder_y_position , 0)) # remember tranlsation always happens in global coordinates so x, y, z.
            girders = girders.union(girder)

        bridge_core = deck_slab.union(girders)
        
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
        # Railings run the full deck length, over the abutments too.
        deck_length = self.config.total_length_m
        road_z = self.road_z
        

        num_of_poles = max(2, int(deck_length / (railing_pole_distance)))
        

        railing_poles = cq.Workplane("XY")

        # we can take the distance between the 1st pole and the last pole to get the total length of the bar which would be less than the span length. 
        # easy way is to just minus the distance of half poles from both sides.


        for i in range(num_of_poles):
            pole_x_position = - ( railing_pole_distance * (num_of_poles - 1) /2) + railing_pole_distance * i
            pole = cq.Workplane("XY").box(railing_pole_side_length, railing_pole_side_length, railing_pole_height, centered=(True, True, False)).translate((pole_x_position, self.config.width_m / 2 - railing_pole_side_length / 2, road_z))
            railing_poles = railing_poles.union(pole)

        
        railing_poles_mirror = railing_poles.mirror("XZ")
        railing_poles = railing_poles.union(railing_poles_mirror)


        # now we add railing bars between the poles

        num_of_bars = 3
        distance_between_bars = railing_pole_height / num_of_bars
        bars = cq.Workplane("XY")
        for i in range(num_of_bars):
            bar_z_position = distance_between_bars * (i + 1)
            bar = cq.Workplane("XY").box(deck_length - railing_pole_distance, railing_pole_side_length, railing_pole_side_length, centered=(True, True, True)).translate((0, self.config.width_m / 2 - railing_pole_side_length / 2, bar_z_position  + road_z))
            bars = bars.union(bar)
        
        bars_mirror = bars.mirror("XZ")
        bars = bars.union(bars_mirror)

        return railing_poles.union(bars)

    def compute_pier_positions_along_length(self) -> list[float]:
        
        # Piers sit at the boundaries of the sampled span layout.
        layout = getattr(self.config, "span_layout_m", None)
        if self.config.num_spans <= 1 or not layout:
            return []
        x, positions = -self.config.inner_deck_length_m / 2.0, []
        for span in layout[:-1]:
            x += span
            positions.append(round(x, 3))
        return positions

    def make_piers(self) -> Optional[cq.Workplane]:

        if self.config.total_piers <= 0 or not self.config.pier_type:
            return None

        pier_positions_x = self.compute_pier_positions_along_length()
        if not pier_positions_x:
            return None

        # Columns and abutments both stand on z_ground.
        z_ground = self.z_ground
        z_soffit = self.z_soffit

        if self.config.pier_type == "wall":
            # A wall pier carries the deck directly: no cap, no columns.
            wall_thickness = 2 * self.config.radius_of_pier
            wall_width_y = max(1.0, self.config.width_m - 2.0)
            return (
                cq.Workplane("XY")
                .pushPoints([(x, 0) for x in pier_positions_x])
                .box(wall_thickness, wall_width_y, z_soffit - z_ground,
                     centered=(True, True, False))
                .translate((0, 0, z_ground))
            )

        if self.config.pier_type == "multicolumn":

            # Multi column piers require a pier cap
            piers = cq.Workplane("XY")
            cap_height = 0.5
            cap_thickness = 2 * self.config.radius_of_pier * 1.4

            num_of_piers_y = self.config.number_of_piers_across_width
            blades = self.config.pier_cross_section == "trapezoidal"
            if blades:
                # blades at the centres of equal strips across the width
                strip = self.config.width_m / num_of_piers_y
                pier_positions_y = [-self.config.width_m / 2 + strip * (j + 0.5)
                                    for j in range(num_of_piers_y)]
                cap_height = 0.0      # no cap: the blade meets the soffit
            else:
                # Columns stop short of the deck edge; the deck cantilevers past them.
                edge = min(1.2, max(0.3, self.config.width_m * 0.08))
                usable = max(0.5, self.config.width_m - 2 * edge)
                if num_of_piers_y == 1:
                    pier_positions_y = [0.0]
                else:
                    step = usable / (num_of_piers_y - 1)
                    pier_positions_y = [-usable / 2 + step * j for j in range(num_of_piers_y)]

            # Columns run from the ground to the underside of the cap.
            column_height = max(0.3, (z_soffit - cap_height) - z_ground)

            for pier_position_x in pier_positions_x:
                for pier_position_y in pier_positions_y:
                    pier = self.make_column(column_height)
                    piers = piers.union(pier.translate((pier_position_x, pier_position_y, z_ground)))

                if cap_height > 0:
                    pier_cap = self.make_prismatic_pier_caps(cap_height, cap_thickness)
                    piers = piers.union(pier_cap.translate((pier_position_x, 0, z_soffit - cap_height)))

            return piers

        return None

    def make_column(self, height: float) -> cq.Workplane:
        """One pier column standing on z = 0 of its workplane."""
        radius = self.config.radius_of_pier
        shape = self.config.pier_cross_section

        if shape == "cylindrical":
            return cq.Workplane("XY").circle(radius).extrude(height)

        if shape == "trapezoidal":
            return self.make_trapezoidal_blade(height)

        side = 2 * radius
        return cq.Workplane("XY").rect(side, side).extrude(height)

    def make_trapezoidal_blade(self, height: float) -> cq.Workplane:
        """Blade column, narrow at the ground and wider at the deck."""
        thickness = 2 * self.config.radius_of_pier
        strip = self.config.width_m / max(1, self.config.number_of_piers_across_width)
        top = min(TRAPEZOID_TOP_OVER_THICKNESS * thickness, 0.8 * strip)
        top = max(top, 1.5 * thickness)
        base = max(top / TRAPEZOID_TOP_OVER_BASE, 1.15 * thickness)
        # rotate so the blade lies across the road
        return (
            cq.Workplane("XY")
            .slot2D(base, thickness, angle=90)
            .workplane(offset=height)
            .slot2D(top, thickness, angle=90)
            .loft(ruled=True)
        )

    def make_prismatic_pier_caps(self, cap_height: float, cap_thickness: float) -> Optional[cq.Workplane]:
     
        """ This makes the prismatic pier caps """
        

        cap_width = self.config.width_m     # width along traffic
        pier_cap = cq.Workplane("YZ").box(cap_width, cap_height, cap_thickness, centered=(True, False, True))
        return pier_cap
        
    def abutment_style(self) -> str:
        """Wingwall abutment for single span, spill-through for multi span."""
        style = getattr(self.config, "abutment_style", "") or ""
        if style:
            return style
        return "spill_through" if self.config.num_spans > 1 else "wingwall"

    def make_abutments(self) -> cq.Workplane:
        style = self.abutment_style()
        if style == "wingwall":
            return self.make_wingwall_abutment()
        if style == "spill_through":
            return self.make_spill_through_abutment()
        if style == "prismatic":
            return self.make_prismatic_abutment()
        raise ValueError(f"Invalid abutment style: {style}")

    @property
    def z_deck_underside(self) -> float:
        """Height where the abutment top meets the deck."""
        return self.z_soffit if self.config.bridge_type == "slab" else 0.0

    def _xz_prism(self, points, width_y: float, y_centre: float = 0.0) -> cq.Workplane:
        """Closed polygon drawn in elevation (x, z), extruded symmetrically in y."""
        return (
            cq.Workplane("XZ")
            .polyline(points).close()
            .extrude(width_y / 2.0, both=True)
            .translate((0, y_centre, 0))
        )

    def make_wingwall_abutment(self) -> cq.Workplane:
        """Breast wall across the opening plus triangular wing walls."""
        c = self.config
        ext = max(c.wing_wall_extension_m, 0.3)
        thk = c.wing_wall_thickness
        x_face = c.inner_deck_length_m / 2.0
        x_end = x_face + ext
        z_bot = self.z_ground - 0.5          # a little embedment
        z_top = self.z_deck_underside
        height = z_top - z_bot
        breast = min(ext, max(0.6, min(1.2, 0.35 * ext)))
        tip = min(0.5, 0.15 * height)        # the wing wall ends in a short nose

        one_end = self._xz_prism(
            [(x_face, z_bot), (x_face + breast, z_bot), (x_face + breast, z_top),
             (x_face, z_top)], c.width_m)
        if ext > breast + 0.2:
            wing = [(x_face, z_bot), (x_end, z_top - tip), (x_end, z_top), (x_face, z_top)]
            for sy in (1, -1):
                one_end = one_end.union(
                    self._xz_prism(wing, thk, sy * (c.width_m / 2.0 - thk / 2.0)))
        return one_end.union(one_end.mirror("YZ"))

    def make_spill_through_abutment(self) -> cq.Workplane:
        """Bank seat under the deck end and a sloped revetment below it."""
        c = self.config
        x_face = c.inner_deck_length_m / 2.0
        x_end = x_face + max(c.wing_wall_extension_m, 0.3)
        z_top = self.z_deck_underside
        z_ground = self.z_ground
        height = z_top - z_ground
        seat = min(max(SPILL_SEAT_HEIGHT[0], 0.25 * height), SPILL_SEAT_HEIGHT[1], 0.5 * height)
        z_seat = min(self.z_soffit, z_top) - seat
        drop = max(z_seat - z_ground, 0.0)

        layout = getattr(c, "span_layout_m", None) or []
        ends = (layout[-1], layout[0]) if len(layout) > 1 else (c.inner_deck_length_m / 2.0,) * 2
        slope = getattr(c, "abutment_slope", 2.0) or 2.0

        parts = None
        for sx, end_span in zip((1, -1), ends):
            # Keep the toe clear of the first pier (column half-thickness + 0.5 m).
            reach = min(slope * drop, max(0.5, end_span - c.radius_of_pier - 0.5))
            if drop > SPILL_REVETMENT + 0.2:
                # revetment slab on the slope
                run = SPILL_REVETMENT * reach / drop
                z_under = z_seat - SPILL_REVETMENT
                pts = [(x_end, z_top), (x_face, z_top), (x_face, z_seat),
                       (x_face - reach, z_ground), (x_face - reach + run, z_ground),
                       (x_face, z_under), (x_end, z_under)]
            else:
                pts = [(x_end, z_top), (x_face, z_top), (x_face, z_seat),
                       (x_face - reach, z_ground), (x_end, z_ground)]
            body = self._xz_prism(pts, c.width_m)
            if sx < 0:
                body = body.mirror("YZ")
            parts = body if parts is None else parts.union(body)
        return parts

    def make_prismatic_abutment(self) -> cq.Workplane:
        """Back wall across the deck end plus a straight wing wall each side."""
        c = self.config
        ext = c.wing_wall_extension_m
        thk = c.wing_wall_thickness
        x_face = c.inner_deck_length_m / 2.0

        # up to the deck underside
        z_bot = self.z_ground - 0.5          # a little embedment
        height = self.z_soffit - z_bot

        # Capped so the abutment never protrudes past the end of the deck.
        back_thk = min(ext, max(0.6, ext * 0.35), 2.5)
        abutments = (
            cq.Workplane("XY")
            .pushPoints([(x_face + back_thk / 2.0, 0), (-x_face - back_thk / 2.0, 0)])
            .box(back_thk, c.width_m, height, centered=(True, True, False))
            .translate((0, 0, z_bot))
        )

        if ext > back_thk + 0.2:
            # wing walls
            wing_len = ext - back_thk
            x_c = x_face + back_thk + wing_len / 2.0
            y_c = c.width_m / 2.0 - thk / 2.0
            abutments = abutments.union(
                cq.Workplane("XY")
                .pushPoints([(sx * x_c, sy * y_c) for sx in (1, -1) for sy in (1, -1)])
                .box(wing_len, thk, height, centered=(True, True, False))
                .translate((0, 0, z_bot))
            )

        return abutments

    def make_ground(self) -> Optional[cq.Workplane]:
        """Ground, backfill and approach embankment. Occluder only."""
        c = self.config
        z_ground = self.z_ground
        rise = -z_ground                  # bed up to deck soffit level
        margin = getattr(c, "ground_extra_m", 12.0)
        batter = 1.5 * rise               # 1:1.5 side slope

        # One plate over the whole footprint, so the plan has no corner gaps.
        ground = cq.Workplane("XY").box(
            c.total_length_m + 2 * margin,
            c.width_m + 2 * (margin + batter),
            0.8, centered=(True, True, False),
        ).translate((0, 0, z_ground - 0.8))

        x_face = c.inner_deck_length_m / 2.0
        half_len = c.total_length_m / 2.0
        fill_len = half_len - x_face
        if fill_len > 0.3 and rise > 0.2:
            # backfill between the wing walls
            inset = c.wing_wall_thickness * 1.2
            ground = ground.union(
                cq.Workplane("XY")
                .pushPoints([(x_face + fill_len / 2, 0), (-x_face - fill_len / 2, 0)])
                .box(fill_len, max(1.0, c.width_m - 2 * inset), rise,
                     centered=(True, True, False))
                .translate((0, 0, z_ground))
            )

        if rise > 0.2:
            # approach embankment with 1:1.5 side slopes
            for sign in (1, -1):
                ground = ground.union(
                    cq.Workplane("XY")
                    .rect(margin, c.width_m + 2 * batter)
                    .workplane(offset=rise)
                    .rect(margin, c.width_m)
                    .loft(ruled=True)
                    .translate((sign * (half_len + margin / 2), 0, z_ground))
                )
        return ground

    def build_bridge(self, with_components: bool = False, include_ground: bool = True) -> cq.Workplane | Dict[str, cq.Workplane]:
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
            "railings": self.make_railings(),
            "piers": self.make_piers(),
            "abutments": self.make_abutments(),
            # ground is not part of the preview solid
            "ground": self.make_ground() if include_ground else None,
        }

        filtered = {name: solid for name, solid in components.items() if solid is not None}

        bridge: cq.Workplane | None = None
        for name, solid in filtered.items():
            if name in NON_STRUCTURE_COMPONENTS:
                continue
            bridge = solid if bridge is None else bridge.union(solid)

        if bridge is None:
            raise ValueError("No bridge components were generated; check configuration.")

        if with_components:
            return filtered, bridge
        else:
            return bridge

        
       

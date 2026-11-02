from .helios_config import (
    WIDTH_OFFSET, LENGTH_OFFSET,
    Z_LEG_1_2_ABOVE_DECK, Z_LEG_3_4_ABOVE_DECK,
    Z_LEG_5_6_ABOVE_GROUND, Z_LEG_7_8_ABOVE_DECK,
    Z_LEG_1_2, Z_LEG_3_4, Z_LEG_5_6, Z_LEG_7_8,
)


def bridge_levels(bridge):
    """Road level and ground level for a bridge record."""
    depth = bridge.get('depth_of_girder')
    clearance = bridge.get('bridge_clearance_height')
    if depth is None or clearance is None:
        return None, None
    road_z = bridge.get('deck_thickness', 0.3)
    return road_z, -depth - clearance


def calculate_scanner_positions(bridge):
    """Calculate scanner positions for all 8 legs based on bridge dimensions.

    Args:
        bridge: Dictionary containing bridge parameters

    Returns:
        Dictionary with positions for leg1 through leg8
    """
    width = bridge['width_m']
    length = bridge['total_length_m']

    road_z, ground_z = bridge_levels(bridge)
    if road_z is None:
        # Old-format record: fall back to the original absolute heights.
        z12, z34, z56, z78 = Z_LEG_1_2, Z_LEG_3_4, Z_LEG_5_6, Z_LEG_7_8
    else:
        z12 = road_z + Z_LEG_1_2_ABOVE_DECK
        z34 = road_z + Z_LEG_3_4_ABOVE_DECK
        z56 = ground_z + Z_LEG_5_6_ABOVE_GROUND
        z78 = road_z + Z_LEG_7_8_ABOVE_DECK

    # Leg 1 & 2: Left and right sides
    leg1_y = (width / 2) + WIDTH_OFFSET
    leg2_y = -(width / 2) - WIDTH_OFFSET

    # Leg 3 & 4: Front and back along length
    leg3_x = (length / 2) + LENGTH_OFFSET
    leg4_x = -(length / 2) - LENGTH_OFFSET

    # Leg 5 & 6: Below the deck, standing on the bank
    leg5_x = leg4_x / 3  # 1/3 of leg 4 (negative)
    leg5_y = 2 * leg1_y  # 2x of leg 1
    leg6_x = leg3_x / 3  # 1/3 of leg 3
    leg6_y = 2 * leg2_y  # 2x of leg 2

    # Leg 7 & 8: Front and back, elevated
    leg7_x = leg3_x
    leg8_x = leg4_x

    return {
        'leg1': {'x': 0, 'y': leg1_y, 'z': z12},
        'leg2': {'x': 0, 'y': leg2_y, 'z': z12},
        'leg3': {'x': leg3_x, 'y': 0, 'z': z34},
        'leg4': {'x': leg4_x, 'y': 0, 'z': z34},
        'leg5': {'x': leg5_x, 'y': leg5_y, 'z': z56},
        'leg6': {'x': leg6_x, 'y': leg6_y, 'z': z56},
        'leg7': {'x': leg7_x, 'y': 0, 'z': z78},
        'leg8': {'x': leg8_x, 'y': 0, 'z': z78},
    }

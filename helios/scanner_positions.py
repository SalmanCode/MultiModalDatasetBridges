from config import WIDTH_OFFSET, LENGTH_OFFSET, Z_LEG_1_2, Z_LEG_3_4, Z_LEG_5_6, Z_LEG_7_8


def calculate_scanner_positions(bridge):
    """Calculate scanner positions for all 8 legs based on bridge dimensions.
    
    Args:
        bridge: Dictionary containing bridge parameters (width_m, total_length_m)
    
    Returns:
        Dictionary with positions for leg1 through leg8
    """
    width = bridge['width_m']
    length = bridge['total_length_m']
    
    # Leg 1 & 2: Left and right sides (x=0, z=Z_LEG_1_2)
    leg1_y = (width / 2) + WIDTH_OFFSET
    leg2_y = -(width / 2) - WIDTH_OFFSET
    
    # Leg 3 & 4: Front and back along length (y=0, z=Z_LEG_3_4)
    leg3_x = (length / 2) + LENGTH_OFFSET
    leg4_x = -(length / 2) - LENGTH_OFFSET
    
    # Leg 5 & 6: Below at z=-5, x=1/3 of leg3/4, y=2x of leg1/2
    leg5_x = leg4_x / 3  # 1/3 of leg 4 (negative)
    leg5_y = 2 * leg1_y  # 2x of leg 1
    leg6_x = leg3_x / 3  # 1/3 of leg 3
    leg6_y = 2 * leg2_y  # 2x of leg 2
    
    # Leg 7 & 8: Front and back, y=0, z=10
    leg7_x = leg3_x
    leg8_x = leg4_x
    
    return {
        'leg1': {'x': 0, 'y': leg1_y, 'z': Z_LEG_1_2},
        'leg2': {'x': 0, 'y': leg2_y, 'z': Z_LEG_1_2},
        'leg3': {'x': leg3_x, 'y': 0, 'z': Z_LEG_3_4},
        'leg4': {'x': leg4_x, 'y': 0, 'z': Z_LEG_3_4},
        'leg5': {'x': leg5_x, 'y': leg5_y, 'z': Z_LEG_5_6},
        'leg6': {'x': leg6_x, 'y': leg6_y, 'z': Z_LEG_5_6},
        'leg7': {'x': leg7_x, 'y': 0, 'z': Z_LEG_7_8},
        'leg8': {'x': leg8_x, 'y': 0, 'z': Z_LEG_7_8},
    }

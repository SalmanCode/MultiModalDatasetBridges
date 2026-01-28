import json

# Read bridge_1 data
with open('Generated_Bridges/bridge_summary.json', 'r') as f:
    bridges = json.load(f)

bridge = bridges[0]  # bridge_1
width = bridge['width_m']
length = bridge['total_length_m']

WIDTH_OFFSET = 5
LENGTH_OFFSET = 5
Z_LEG_1_2 = 3
Z_LEG_3_4 = 1
Z_LEG_5_6 = -5
Z_LEG_7_8 = 10

print(f"Bridge: {bridge['bridge_id']}")
print(f"Dimensions: Length={length}m, Width={width}m\n")

# Calculate positions
leg1_y = (width / 2) + WIDTH_OFFSET
leg2_y = -(width / 2) - WIDTH_OFFSET
leg3_x = (length / 2) + LENGTH_OFFSET
leg4_x = -(length / 2) - LENGTH_OFFSET
leg5_x = leg4_x / 3
leg5_y = 2 * leg1_y
leg6_x = leg3_x / 3
leg6_y = 2 * leg2_y
leg7_x = leg3_x
leg8_x = leg4_x

print('Scanner Positions:')
print(f'Leg 1 (Left):   x={0:7.1f}, y={leg1_y:7.1f}, z={Z_LEG_1_2:7.1f}')
print(f'Leg 2 (Right):  x={0:7.1f}, y={leg2_y:7.1f}, z={Z_LEG_1_2:7.1f}')
print(f'Leg 3 (Front):  x={leg3_x:7.1f}, y={0:7.1f}, z={Z_LEG_3_4:7.1f}')
print(f'Leg 4 (Back):   x={leg4_x:7.1f}, y={0:7.1f}, z={Z_LEG_3_4:7.1f}')
print(f'Leg 5 (Below1): x={leg5_x:7.1f}, y={leg5_y:7.1f}, z={Z_LEG_5_6:7.1f}')
print(f'Leg 6 (Below2): x={leg6_x:7.1f}, y={leg6_y:7.1f}, z={Z_LEG_5_6:7.1f}')
print(f'Leg 7 (Top1):   x={leg7_x:7.1f}, y={0:7.1f}, z={Z_LEG_7_8:7.1f}')
print(f'Leg 8 (Top2):   x={leg8_x:7.1f}, y={0:7.1f}, z={Z_LEG_7_8:7.1f}')

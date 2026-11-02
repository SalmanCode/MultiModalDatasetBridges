# Configuration parameters for HELIOS scanner positioning

WIDTH_OFFSET = 7  # meters to add/subtract from width for legs 1 & 2
LENGTH_OFFSET = 25 # meters to add/subtract from length for legs 3 & 4

# scanner heights relative to the deck or the ground
TRIPOD_HEIGHT = 1.7        # instrument height above whatever it stands on

Z_LEG_1_2_ABOVE_DECK = 1.3   # legs 1 & 2: on the deck, at the kerb
Z_LEG_3_4_ABOVE_DECK = 1.3   # legs 3 & 4: on the approach road, deck level
Z_LEG_5_6_ABOVE_GROUND = TRIPOD_HEIGHT   # legs 5 & 6: on the bank, under the deck
Z_LEG_7_8_ABOVE_DECK = 8.0   # legs 7 & 8: elevated / mast positions

# fallback absolute heights for old records
Z_LEG_1_2 = 3
Z_LEG_3_4 = 1
Z_LEG_5_6 = -5
Z_LEG_7_8 = 10

"""Fixed supermarket zones copied from the official Server referee config."""

import math

# Server referee uses MuJoCo time, not wall time.
TIME_LIMIT_S = 420.0

# Carry-out and placement thresholds used by the built-in referee.
CARRY_OUT_DIST_M = 0.20
SETTLE_SPEED_M_S = 0.02
DROP_Z_M = 0.30
UPRIGHT_TOL_DEG = 15.0

PICKING_ZONE = {"x": (-2.5, 2.5), "y": (1.70, 3.25)}
DELIVERY_BASE_ZONE = {"x": (-2.42, -1.46), "y": (-3.88, -2.62)}
DELIVERY_BOX_ZONE = {"x": (-2.42, -1.46), "y": (-3.63, -3.19), "z": (0.74, 1.05)}

# MJCF: body delivery_table at (-1.940, -3.410, 0), site delivery_target z=0.807.
DELIVERY_TABLE_XY = (-1.940, -3.410)
DELIVERY_TARGET_XYZ = (-1.940, -3.410, 0.807)

# Approach the table from the north (+Y) so the base stays inside delivery_base.
DELIVERY_APPROACH_XY = (-1.940, -2.90)
DELIVERY_FACE_YAW = -math.pi / 2.0
# South gap is too tight for the base. Sim logs hit the divider at y≈-2.80
# and y≈-3.10; the south wall is around y=-3.45. Do not westbound here.
SOUTH_CROSS_Y = -3.22

# Yellow picking aisle (y=1.70 / 3.25) and the official shelf-approach lane.
# Spawn is the delivery area; the known clear path is northeast into the aisle,
# then west along the yellow lines. Face slightly east of north (official
# grasp yaw) so the head camera looks at the shelf face, not the aisle gap.
# Turn west *south* of the shelf face: (1.92, 2.475) is too close to the racks.
SHELF_AISLE_Y = 2.475
SHELF_CORNER_Y = 2.00
SHELF_AISLE_ENTRY_XY = (1.92, SHELF_AISLE_Y)
SHELF_CORNER_XY = (1.92, SHELF_CORNER_Y)
SHELF_APPROACH_XY = (0.852, SHELF_AISLE_Y)
SHELF_FACE_YAW = math.pi / 2.0 - math.radians(11.0)
# Westbound lane: south of the racks, north of the divider. y=2.00 points
# the lidar at the wall and triggers reverse in front of the shelf. The slab
# north tip is y=1.90; 8x westbound at y=2.20 dipped into that corner.
WEST_LANE_Y = 2.32
# Inner west face of the center wall. Boxes sit on the outer (x≈-1.9) side
# of the left corridor; hugging this rail is the smooth southbound lane.
HUG_WEST_X = -0.40
SOUTH_PEEL_Y = -2.70
# East corridor north to the yellow lane, then west. y=2.00 is the divider
# face — westbound there drives into the slab (logs turned at y≈1.68).
ROUTE_DELIVERY_TO_SHELF = (
    (SHELF_CORNER_XY[0], WEST_LANE_Y),
    (SHELF_APPROACH_XY[0], WEST_LANE_Y),
)
# Cross the north gap, hug the divider south, then peel to the table.
ROUTE_SHELF_TO_DELIVERY = (
    (HUG_WEST_X, WEST_LANE_Y),
    (HUG_WEST_X, SOUTH_PEEL_Y),
    DELIVERY_APPROACH_XY,
)
# East stub in front of the north racks. Do not keep driving +Y here.
EAST_STUB_X_MIN = 1.35
EAST_STUB_Y_MAX = 2.50
# Shelf face. Any x this far north is in the racks, not just the east stub.
NORTH_RACK_Y = 2.48
# South-east corner: do not keep driving -Y into the south wall.
SOUTH_EAST_X_MIN = 1.35
SOUTH_EAST_Y_MIN = -3.45
# Center divider. Westbound in this band hits the wall; go around south.
CENTER_WALL_X = (0.20, 1.45)
CENTER_WALL_Y = (-3.08, 1.90)

# Loose footprint used to keep lidar detours inside the supermarket.
NAV_BOUNDS = {"x": (-2.55, 2.35), "y": (-3.85, 3.30)}


def point_in_zone(point, zone) -> bool:
    x, y = float(point[0]), float(point[1])
    ok = zone["x"][0] <= x <= zone["x"][1] and zone["y"][0] <= y <= zone["y"][1]
    if "z" in zone:
        if len(point) < 3:
            return False
        ok = ok and zone["z"][0] <= float(point[2]) <= zone["z"][1]
    return ok


def in_picking_zone(xy) -> bool:
    return point_in_zone(xy, PICKING_ZONE)


def in_delivery_base(xy) -> bool:
    return point_in_zone(xy, DELIVERY_BASE_ZONE)


def in_delivery_box(xyz) -> bool:
    return point_in_zone(xyz, DELIVERY_BOX_ZONE)


def in_east_shelf_stub(x: float, y: float) -> bool:
    return float(x) >= EAST_STUB_X_MIN and float(y) >= EAST_STUB_Y_MAX


def in_south_east_stub(x: float, y: float) -> bool:
    return float(x) >= SOUTH_EAST_X_MIN and float(y) <= SOUTH_EAST_Y_MIN


def in_north_racks(x: float, y: float) -> bool:
    return NAV_BOUNDS["x"][0] <= float(x) <= NAV_BOUNDS["x"][1] and float(y) >= NORTH_RACK_Y


def in_center_wall_band(x: float, y: float) -> bool:
    return (
        CENTER_WALL_X[0] <= float(x) <= CENTER_WALL_X[1]
        and CENTER_WALL_Y[0] <= float(y) <= CENTER_WALL_Y[1]
    )


def near_divider_nw_corner(x: float, y: float) -> bool:
    """Inflated north-west tip of the center wall. Southbound here wedges the base."""
    return -0.18 <= float(x) <= 0.55 and 1.40 <= float(y) <= 2.22

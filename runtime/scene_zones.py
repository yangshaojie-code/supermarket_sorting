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

# Yellow picking aisle (y=1.70 / 3.25) and the official shelf-approach lane.
# Spawn is the delivery area; the known clear path is northeast into the aisle,
# then west along the yellow lines. Face slightly east of north (official
# grasp yaw) so the head camera looks at the shelf face, not the aisle gap.
SHELF_AISLE_Y = 2.475
SHELF_AISLE_ENTRY_XY = (1.92, SHELF_AISLE_Y)
SHELF_APPROACH_XY = (0.852, SHELF_AISLE_Y)
SHELF_FACE_YAW = math.pi / 2.0 - math.radians(11.0)
ROUTE_DELIVERY_TO_SHELF = (SHELF_AISLE_ENTRY_XY, SHELF_APPROACH_XY)


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

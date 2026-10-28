"""Small map-distance helpers (no external libraries).

All distances are in metres. Coordinates are (latitude, longitude) in degrees,
except route geometries, which follow GeoJSON order: [longitude, latitude].
"""

import math

EARTH_RADIUS_M = 6_371_000
COMPASS_POINTS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance between two points on the Earth."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lng2 - lng1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_name(lat1: float, lng1: float, lat2: float, lng2: float) -> str:
    """Compass direction from point 1 to point 2, such as "northeast"."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lng2 - lng1)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    degrees = (math.degrees(math.atan2(x, y)) + 360) % 360
    return COMPASS_POINTS[int((degrees + 22.5) // 45) % 8]


def point_to_segment_m(lat: float, lng: float, lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    """Shortest distance from a point to the line segment A-B.

    Uses a flat local map around the point, which is accurate to well under
    a metre over the few kilometres that matter for Metro Manila streets.
    """
    metres_per_degree_lat = 111_320
    metres_per_degree_lng = 111_320 * math.cos(math.radians(lat))

    ax, ay = (lng_a - lng) * metres_per_degree_lng, (lat_a - lat) * metres_per_degree_lat
    bx, by = (lng_b - lng) * metres_per_degree_lng, (lat_b - lat) * metres_per_degree_lat

    segment_x, segment_y = bx - ax, by - ay
    segment_length_squared = segment_x**2 + segment_y**2
    if segment_length_squared == 0:
        return math.hypot(ax, ay)

    # Position of the closest point along the segment, from 0 (A) to 1 (B).
    position = max(0.0, min(1.0, -(ax * segment_x + ay * segment_y) / segment_length_squared))
    return math.hypot(ax + position * segment_x, ay + position * segment_y)


def circle_polygon(lat: float, lng: float, radius_m: float, sides: int = 12) -> list[list[float]]:
    """A closed ring of [lng, lat] points approximating a circle (for "avoid this area")."""
    ring = []
    for step in range(sides):
        angle = 2 * math.pi * step / sides
        d_lat = (radius_m * math.cos(angle)) / 111_320
        d_lng = (radius_m * math.sin(angle)) / (111_320 * math.cos(math.radians(lat)))
        ring.append([round(lng + d_lng, 6), round(lat + d_lat, 6)])
    ring.append(ring[0])
    return ring
